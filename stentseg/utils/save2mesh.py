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
    mesh = create_mesh(model, 0.4)  # Param is thickness (~0.75mm diam)
    mesh._vertices[:,-1] = mesh._vertices[:,-1]*-1 # flip z, negative in original dicom
    vv.meshWrite(os.path.join(savedir, filename),mesh)



if __name__ == '__main__':
    import os
    import visvis as vv
    from stentseg.utils.datahandling import select_dir, loadvol, loadmodel

    # Select the ssdf basedir
    basedir = select_dir(os.getenv('LSPEAS_BASEDIR', ''),
                        r'D:\LSPEAS\LSPEAS_ssdf',
                        r'F:\LSPEAS_ssdf_backup',r'G:\LSPEAS_ssdf_backup')
    
    # Dir to save mesh
    savedir = r'C:\Users\Maaike\Desktop' 
    
    # Select dataset to register
    ptcode = 'LSPEAS_003'
    ctcode = 'discharge'
    cropname = 'ring'
    modelname = 'modelavgreg'
    
    # Save mesh
    model2mesh(basedir,savedir,ptcode,ctcode,cropname,modelname=modelname)
    print('mesh saved to %s' % savedir )
    
    
    