""" Store model as mesh
Stl format. Obj also an option.
"""

def model2mesh(basedir,savedir,ptcode,ctcode,cropname,modelname='modelavgreg'):
    from stentseg.stentdirect.stentgraph import create_mesh
    
    # formats that can be saved to: .obj .stl .ssdf .bsdf 
    filename = '%s_%s_%s_%s.stl' % (ptcode, ctcode, cropname, 'model')
    # Load the stent model and mesh
    s = loadmodel(basedir, ptcode, ctcode, cropname, modelname)
    model = s.model
    mesh = create_mesh(model, 0.4)  # Param is thickness (with 0.4 -> ~0.75mm diam)
    mesh._vertices[:,-1] *= -1 # flip z, negative in original dicom
    mesh._normals[:,-1] *= -1  # flip also normals to change front face and back face along
    vv.meshWrite(os.path.join(savedir, filename),mesh)
    

if __name__ == '__main__':
    import os
    import visvis as vv
    from stentseg.utils.datahandling import select_dir, loadvol, loadmodel

    # Select the ssdf basedir
    basedir = select_dir(r'D:\LSPEAS\LSPEAS_ssdf',
                        r'F:\LSPEAS_ssdf_backup',r'G:\LSPEAS_ssdf_backup')
    
    # Dir to save mesh
    savedir = select_dir(r'D:\LSPEAS\LSPEAS_mesh_ring', r'C:\Users\Maaike\Desktop\LSPEAS_ring_mesh [copy C to PC]')
    
    # Select dataset to register
    # ptcodes = ['LSPEAS_001','LSPEAS_002','LSPEAS_003','LSPEAS_005','LSPEAS_008',
    #             'LSPEAS_009','LSPEAS_011','LSPEAS_015','LSPEAS_017','LSPEAS_018',
    #             'LSPEAS_019','LSPEAS_020','LSPEAS_021','LSPEAS_022','LSPEAS_025',
    #             'LSPEAS_023']
    ptcodes = ['LSPEAS_004']
    ctcode = '12months'
    cropname = 'ring'
    modelname = 'modelavgreg'
    
    for ptcode in ptcodes:
        # Save mesh
        model2mesh(basedir,savedir,ptcode,ctcode,cropname,modelname=modelname)
        print('mesh saved to %s' % savedir )
    
    
    