import numpy as np


def calculate_runoff_maps(htop,slopex,slopey,mannings,nx,ny,dx,dy,nt,mask,KWE='UP',large_array=True,epsilon = 1E-7):

    """
    Calculate runoff for each pixel following Rcalc in Priority flow.
    Two options of flow calc (UP and STAN) and 2 options regarding problem dimensions: using time loops of large matrix)

    2D arrays are : htop, slopex, slopey and mannings. first dimension is x (West->East), second is y (South->North). So lower left is [0,0] and upper right is [nx,ny]

    htop is surface pressure
    options are:
    - KWE: 'UP' stands for former overland calculation where the slopes are given at the cell centers and upwinded using fluxes
    - KWE: 'Stan' stands for the new OverlandKinematic formulation where the slopes are given at the cell interfaces and upwinded using h
    - large_array: if htop is very large (nx,ny,nt), this is the default, and time loop are being used. Otherwise, a direct 3D matrix method is used (for KWE='UP')

    returns:
    outflow: a 3D array (NX,NY,NT)

    The comments and R code from Priority flow are included for tracking purposes.

    TODO: implement 3D matrix operations for KWE = 'Stan'
    """

    outflow = np.zeros((nx, ny, nt))

    if KWE == 'UP':

        if not large_array:
            # if working with large NX NY NT array is doable:
            # Units should be l3/t
            qx = -np.sign(np.repeat(slopex[:, :, np.newaxis], nt, axis=2)) * np.abs(np.repeat(slopex[:, :, np.newaxis], nt, axis=2))**0.5/np.repeat(mannings[:, :, np.newaxis], nt, axis=2) * np.maximum(htop, np.zeros((nx, ny, nt)))**(5/3) * dy
            # 100s for 2 years oueme domain (144x144)
            qeast = np.maximum(qx[0:(nx-1), :, :], np.zeros((nx-1, ny, nt))) - np.maximum(-qx[1: nx, :, :], np.zeros((nx-1, ny, nt)))
            qeast = np.concatenate((-np.maximum(-np.reshape(qx[0, :, :], (1, ny, nt)), np.zeros((1, ny, nt))), qeast), axis=0)
            qeast = np.concatenate((qeast, np.maximum(np.reshape(qx[nx-1, :, :], (1, ny, nt)), np.zeros((1, ny, nt)))), axis=0)
            # Units should be l3/t
            qy = -np.sign(np.repeat(slopey[:, :, np.newaxis], nt, axis=2)) * np.abs(np.repeat(slopey[:, :, np.newaxis], nt, axis=2))**0.5/np.repeat(mannings[:, :, np.newaxis], nt, axis=2) * np.maximum(htop, np.zeros((nx, ny, nt)))**(5/3) * dx
            qnorth = np.maximum(qy[:, 0:(ny-1), :], np.zeros((nx, ny-1, nt))) - np.maximum(-qy[:, 1:ny, :], np.zeros((nx, ny-1, nt)))
            qnorth = np.concatenate((-np.maximum(-np.reshape(qy[:, 0, :], (nx, 1, nt)), np.zeros((nx,1,nt))), qnorth), axis=1)
            qnorth = np.concatenate((qnorth, np.maximum(np.reshape(qy[:, ny-1, :],(nx, 1, nt)), np.zeros((nx, 1, nt)))), axis=1)
            outflow = np.maximum(qeast[1:nx+1, :, :],np.zeros((nx, ny, nt))) + np.maximum(-qeast[0:nx, :, :], np.zeros((nx, ny, nt))) + np.maximum(qnorth[:, 1:ny+1, :], np.zeros((nx, ny, nt))) + np.maximum(-qnorth[:, 0:ny, :], np.zeros((nx, ny, nt)))
        else:

            for i in range(nt):
                ptop = htop[:, :, i]
                ptop[ptop < 0] = 0

                #####

                # Calculate fluxes across east and north faces
                # First the x direction

                # Units should be l3/t
                qx = - np.sign(slopex)*np.abs(slopex)**0.5/mannings * ptop**(5/3) * dy

                # Upwinding to get flux across the east face of cells - based in qx[i] if its positive and qx[i+1] if its negative
                # qeast= pmax(qx[1:(nx-1),],zeros[1:(nx-1),]) - pmax(-qx[2:nx,],zeros[2:nx,])
                # TO CHECK max formulaton (axis=0 ?)

                qeast= np.maximum(qx[0:(nx-1),:],np.zeros((nx-1,ny))) - np.maximum(-qx[1:nx,:],np.zeros((nx-1,ny))) # 100s for 2 years oueme domain (144x144)

                # tmp1 =  qx[0:nx-1,:]

                # tmp2 = -qx[1:nx,:]
                # qeast = tmp1.clip(0) - tmp2.clip(0) # 102s for 2 years oueme domain (144x144)
                # qeast = np.where(tmp1>0,tmp1,0) -np.where(tmp2>0,tmp2,0) #  106s for 2 years oueme domain (144x144)

                # adding the left boundary - pressures outside domain are 0 so flux across this boundary only occurs when qx[1] is negative

                # qeast= rbind(-pmax(-qx[1,],0), qeast)

                # TO CHECK:
                qeast=np.concatenate((-np.maximum(-np.reshape(qx[0,:],(1,ny)),np.zeros((1,ny))),qeast),axis=0) 

                # adding the right boundary - pressures outside domain are 0 so flux across this boundary only occurs when qx[nx] is positive
                # qeast= rbind(qeast, pmax(qx[nx,],0))

                qeast=np.concatenate((qeast,np.maximum(np.reshape(qx[nx-1,:],(1,ny)),np.zeros((1,ny)))),axis=0) 

                #####

                # Next the y direction

                # qy= -sign(slopey)*abs(slopey)^0.5/mannings * ptop^(5/3) * dx #Units should be l3/t

                qy = -np.sign(slopey)*np.abs(slopey)**0.5/mannings * ptop**(5/3) * dx #Units should be l3/t

                # Upwinding to get flux across the north face of cells - based in qy[j] if its positive and qy[j+1] if its negative

                # qnorth= pmax(qy[,1:(ny-1)],zeros[,1:(ny-1)]) - pmax(-qy[, 2:ny],zeros[, 2:ny])

                qnorth= np.maximum(qy[:,0:(ny-1)],np.zeros((nx,ny-1))) - np.maximum(-qy[:,1:ny],np.zeros((nx,ny-1)))

                # tmp1 =  qy[:,0:(ny-1)]
                # tmp2 = -qx[:,1:ny]
                # qnorth = tmp1.clip(0) - tmp2.clip(0)
                # qnorth = np.where(tmp1>0,tmp1,0) -np.where(tmp2>0,tmp2,0)

                # adding the bottom - pressures outside domain are 0 so flux across this boundary only occurs when
                # qy[1] is negative
                # qnorth= cbind(-pmax(-qy[,1],0), qnorth)

                # TO CHECK:
                qnorth=np.concatenate((-np.maximum(-np.reshape(qy[:, 0], (nx, 1)), np.zeros((nx, 1))), qnorth), axis=1)

                # adding the right boundary - pressures outside domain are 0 so flux across this boundary only occurs when qx[nx] is positive
                # qnorth= cbind(qnorth, pmax(qy[,ny],0))

                qnorth=np.concatenate((qnorth, np.maximum(np.reshape(qy[:, ny-1], (nx, 1)), np.zeros((nx, 1)))), axis=1)

                # Calculate total outflow

                # Outflow is a positive qeast[i,j] or qnorth[i,j] or a negative qeast[i-1,j], qnorth[i,j-1]
                # outflow=pmax(qeast[2:(nx+1),],zeros) + pmax(-qeast[1:nx,], zeros) +

                #     pmax(qnorth[,2:(ny+1)],zeros) + pmax(-qnorth[, 1:ny], zeros)
                outflow[:, :, i] = np.maximum(qeast[1:nx+1, :], np.zeros((nx, ny))) + np.maximum(-qeast[0:nx, :], np.zeros((nx, ny))) + np.maximum(qnorth[:, 1:ny+1], np.zeros((nx, ny))) + np.maximum(-qnorth[:, 0:ny], np.zeros((nx, ny)))

                # tmp1 = qeast[1:nx+1,:]
                # tmp2 = -qeast[0:nx,:]
                # tmp3 = qnorth[:,1:ny+1]
                # tmp4 = -qnorth[:,0:ny]
                # outflow[:,:,i] =tmp1.clip(0) + tmp2.clip(0) + tmp3.clip(0) + tmp4.clip(0)
                # outflow[:,:,i]  = np.where(tmp1>0,tmp1,0) + np.where(tmp2>0,tmp2,0) + np.where(tmp3>0,tmp3,0) + np.where(tmp4>0,tmp4,0)

    elif KWE == 'Stan':

        for i in range(nt):

            ptop = htop[:, :, i]
            ptop[ptop < 0] = 0

            # Repeat the slopes on the lower and left boundaries that are inside the domain but outside the mask

            # find indices of all cells that are off the mask but have a neigbor to their right that is on the mask
            # fill.left=which((rbind(mask[2:nx,],rep(0,ny)) - mask[1:nx,]) ==1, arr.ind=T)

            fill_left = np.where((np.concatenate((mask[1:nx, :, 0], np.zeros((1, ny))), axis=0) - mask[:, :, 0]) == 1)

            # get the indices of their neigbors to the right

            # fill.left2 = fill.left
            # fill.left2[,1] = fill.left[,1]+1

            fill_left2 = fill_left
            fill_left2[0][:] = fill_left[0][:]+1

            # pad the slopes to the left with their neigboring cells in the mask

            # slopex[fill.left] = slopex[fill.left2]
            slopex[fill_left] = slopex[fill_left2]

            # find indices of all cells that are off the mask but have a neighbor above them that is on the mask
            # fill.down = which((cbind(mask[,2:ny],rep(0,nx)) - mask[,1:ny]) == 1, arr.ind=T)
            fill_down = np.where((np.concatenate((mask[:, 1:ny, 0],np.zeros((nx, 1))), axis=1) - mask[:, :, 0]) == 1)
            # get the indices of their neighbors above
            # fill.down2 = fill.down
            # fill.down2[,2] = fill.down[,2]+1
            fill_down2 = fill_down
            fill_down2[1][:] = fill_down[1][:]+1

            # pad the slopes to below  with their neigboring cells in the mask
            slopey[fill_down] = slopey[fill_down2]

            ####

            # calculate the slope magnitude
            sfmag = np.where((slopex**2 + slopey**2)**0.5 > epsilon, (slopex**2 + slopey**2)**0.5, epsilon)

            # ~ sfmag=np.where((slopex * slopex + slopey * slopey)**0.5>epsilon, (slopex * slopex + slopey * slopey)**0.5, epsilon)

            ###
            # IS THIS NEEDED ? (Basile)

            # For OverlandKinematic slopes are face centered and calculated across the upper and right boundaries
            # (i.e. Z[i+1]-Z[i])
            # For cells on the lower and left boundaries its assumed that the slopes repeat
            # (i.e. repeating the upper and right face boundary for the lower and left for these border cells)

            # slopex.pad=rbind(slopex[1,], slopex)
            # slopey.pad=cbind(slopey[,1], slopey)

            ####

            # upwind the pressure - Note this is for the north and east face of all cells
            # The slopes are calculated across these boundaries so the upper boundary is included in these
            # calculations and the lower and right boundary of the domain will be added later

            # pupwindx = pmax(sign(slopex) * rbind(ptop[2:(nx),], rep(0, ny)),0) + pmax(-sign(slopex) * ptop[1:nx,], 0 )
            # pupwindy = pmax(sign(slopey) * cbind(ptop[, 2:ny], rep(0, nx)),0) + pmax(-sign(slopey) * ptop[, 1:ny], 0)

            pupwindx = np.maximum(np.sign(slopex) * np.concatenate((ptop[1:nx, :], np.zeros((1, ny))), axis=0), np.zeros((nx, ny))) + np.maximum(-np.sign(slopex)*ptop[0:nx, :], np.zeros((nx, ny)))
            pupwindy = np.maximum(np.sign(slopey) * np.concatenate((ptop[:, 1:ny], np.zeros((nx, 1))), axis=1), np.zeros((nx, ny))) + np.maximum(-np.sign(slopey)*ptop[:, 0:ny], np.zeros((nx, ny)))

            # Calculate fluxes across east and north faces

            # First the x direction

            # qeast = -slopex/(sfmag^0.5*mannings) * pupwindx^(5/3) *dy #Units should be l3/t
            # qnorth = -slopey/(sfmag^0.5*mannings) * pupwindy^(5/3) *dx #Units should be l3/t

            qeast = -slopex / ((sfmag**0.5)*mannings) * pupwindx**(5/3) * dy
            qnorth = -slopey / ((sfmag**0.5)*mannings) * pupwindy**(5/3) * dx

            ###

            # Fix the lower x boundary
            # Use the slopes of the first column with the pressures for cell i 

            # qleft = -slopex[1,]/(sfmag[1,]^0.5*mannings)* (pmax(sign(slopex[1,])*ptop[1,],0))^(5/3) * dy
            # qeast = rbind(qleft,qeast)

            qleft = -slopex[0,:]/((sfmag[0,:]**0.5)*mannings) * (np.maximum(np.sign(slopex[0, :]) * ptop[0,:], np.zeros((1, ny))))**(5/3) * dy
            qeast = np.concatenate((qleft,qeast), axis=0)

            ###
            # Fix the lower y boundary
            # Use the slopes of the bottom row with the pressures for cell j

            # qbottom = -slopey[,1]/(sfmag[,1]^0.5*mannings)* (pmax(sign(slopey[,1])*ptop[,1],0))^(5/3) * dx
            # qnorth = cbind(qbottom, qnorth)

            # beware here sfmag[:,0] for instance produce a row vector ! of dimension (144,) and
            # np.zeros((nx,1)).shape = (144, 1) while np.zeros((nx)).shape = (144,)

            qbottom = -slopey[:, 0] / ((sfmag[:, 0]**0.5) * mannings) * (np.maximum(np.sign(slopey[:, 0])*ptop[:, 0],np.zeros((nx))))**(5/3) * dx
            qnorth = np.concatenate((qbottom[:, np.newaxis], qnorth), axis=1)
            outflow[:, :, i] = np.maximum(qeast[1:nx+1, :],np.zeros((nx, ny))) + np.maximum(-qeast[0:nx, :], np.zeros((nx, ny))) + np.maximum(qnorth[:, 1:ny+1], np.zeros((nx, ny))) + np.maximum(-qnorth[:, 0:ny], np.zeros((nx, ny)))

    return outflow
