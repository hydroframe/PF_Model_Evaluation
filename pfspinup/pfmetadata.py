import json
import os.path
from pathlib import Path
import numpy as np
from pfspinup import pfio


class PFMetadata:

    @staticmethod
    def _parse_value(v):

        # TODO: What's with the 'd0' suffix?
        if v.endswith('d0'):
            v = v[:-2]

        if v in ('True', 'False'):
            return eval(v)

        try:
            v = int(v)
        except ValueError:
            try:
                v = float(v)
            except ValueError:
                if ' ' in v:
                    v = [t.strip() for t in v.split(' ')]
                    v = filter(lambda item: item != '', v)
                return v
            else:
                return v
        else:
            return v

    def __init__(self, filename):
        if not os.path.exists(filename):
            raise RuntimeError(f'File {filename} not found')

        self.filename = filename
        self.folder = os.path.abspath(os.path.dirname(filename))
        self.config = json.loads(open(filename, 'r').read())

        # We initialize the 'mask' attribute (an nz-by-nx-by-ny ndarray) early on,
        # since this is used in quite a few places later.
        self.mask = self.input_data('mask', apply_mask=False)

    def __getitem__(self, item):
        value = self.config['inputs']['configuration']['data'][item]
        return self._parse_value(value)

    def _get_absolute_path(self, filename):
        return os.path.join(self.folder, filename)

    def pfb_data(self, filename, apply_mask=True):
        data = pfio.pfread(self._get_absolute_path(filename))
        if apply_mask:
            data[self.mask == 0] = np.nan
        return data

    def input_data(self, which, apply_mask=True):
        input = self.config['inputs'][which]
        assert input['type'] == 'pfb', 'Only pfb input data supported for now'
        assert len(input['data']) == 1, 'Only a single data entry supported for now'
        pfb_file = input['data'][0]['file']
        return self.pfb_data(pfb_file, apply_mask=apply_mask)

    def output_files(self, which, folder=None, index_list=None, ignore_missing=False):
        """
        Return full paths to *.out.*.nnnnn.pfb files, along with other useful information
        :param which: property that saved as time-series pfb files, typically 'pressure' or 'saturation'
        :param folder: Location of time-series pfb files. If None, determined automatically.
        :param index_list: An iterable of index values that the caller is interested in.
            If None, all index values are returned
        :param ignore_missing: Whether to ignore any missing .pfb files or raise an error.
        :return: A 3-tuple of values:
            files: A list of full paths to .pfb files
            index_list: A list of time-index values of files that were returned
            timing_list: A list of timing information of files that were returned
        """
        output = self.config['outputs'][which]
        assert output['type'] == 'pfb', 'Only pfb output data supported for now'
        assert len(output['data']) == 1, 'Only a single data entry supported for now'

        file_series = output['data'][0]['file-series']
        start, end = output['data'][0]['time-range']
        folder = folder or self.folder
        if index_list is None:
            index_list = np.arange(start, end+1)
        else:
            index_list = np.array(index_list)

        files = np.array([os.path.join(folder, file_series % i) for i in index_list])
        files_found = np.vectorize(os.path.exists)(files)
        if not ignore_missing and not all(files_found):
            raise RuntimeError('Some pfb files were not found. Specify ignore_missing=True to ignore these.')

        dump_interval = self['TimingInfo.DumpInterval']
        dt = self['TimingInfo.BaseUnit']
        start_count = self['TimingInfo.StartCount']
        start_time = self['TimingInfo.StartTime']
        out_times = np.array([(i-start_count) * dt * dump_interval + start_time for i in index_list])

        return files[files_found], index_list[files_found], out_times[files_found]

    def nz_list(self, which):
        assert self[f'{which}.Type'] == 'nzList'
        return np.array([self[f'Cell.{i}.{which}.Value'] for i in range(self[f'{which}.nzListNumber'])])

    def get_geom_values(self, which, names_field='Names'):
        d = {}
        for name in self[f'Geom.{which}.{names_field}']:
            assert self[f'Geom.{name}.{which}.Type'] == 'Constant', 'Only constants supported for now'
            d[name] = self[f'Geom.{name}.{which}.Value']
        return d

    def get_geom_by_type(self, typ):
        for geom in self['GeomInput.Names']:
            if self[f'GeomInput.{geom}.InputType'] == typ:
                return geom

    def phase_geom_values(self, phase, attribute_name):
        return {name: self[f'Geom.{name}.{phase}.{attribute_name}'] for name in self[f'Phase.{phase}.GeomNames']}

    def geom_tensors(self, which, axis):
        assert axis in ('X', 'Y', 'Z'), 'axis should be one of X/Y/Z'
        assert self[f'{which}.TensorType'] == 'TensorByGeom', 'Only TensorType = TensorByGeom supported'

        d = {}
        names = self[f'Geom.{which}.TensorByGeom.Names']
        if isinstance(names, str):  # singleton
            d[names] = self[f'Geom.{names}.{which}.TensorVal{axis}']
        else:
            for name in names:
                d[name] = self[f'Geom.{name}.{which}.TensorVal{axis}']
        return d

    def indicator_file(self):
        g = self.get_geom_by_type('IndicatorField')
        return self._get_absolute_path(self[f'Geom.{g}.FileName'])

    def indicator_geom_values(self):
        g = self.get_geom_by_type('IndicatorField')
        d = {}
        for name in self[f'GeomInput.{g}.GeomNames']:
            d[name] = self[f'GeomInput.{name}.Value']
        return d

    def get_single_domain_value(self, which):
        g = self[f'{which}.GeomNames']
        assert type(g) == str, f'Multiple {which}.GeomNames found'
        typ = self[f'{which}.Type']
        if typ == 'Constant':
            try:
                return self[f'Geom.{g}.{which}.Value']
            except KeyError:
                # Some properties (e.g. Mannings have their geom keys reversed)
                return self[f'{which}.Geom.{g}.Value']
        elif typ == 'PFBFile':
            return self.pfb_data(self[f'Geom.{g}.{which}.FileName'])
        elif typ == 'nzList':
            return self.nz_list(which)
        return self[f'Geom.{g}.{which}.Value']

    def dz(self):
        if self['Solver.Nonlinear.VariableDz']:
            dz_scale = self.get_single_domain_value('dzScale')
        else:
            dz_scale = np.ones((self['ComputationalGrid.NZ'],))
        dz_values = dz_scale * self['ComputationalGrid.DZ']
        return dz_values

    def et_flux(self):
        if self['Solver.EvapTransFile']:
            return self.pfb_data(self['Solver.EvapTrans.FileName'])
        elif self['Solver.EvapTransFileTransient']:
            raise NotImplementedError('coming soon..')

    def slope_x(self):
        return self.pfb_data(self['TopoSlopesX.FileName'], apply_mask=False).squeeze(axis=0)

    def slope_y(self):
        return self.pfb_data(self['TopoSlopesY.FileName'], apply_mask=False).squeeze(axis=0)
