
class Nellix2ssdf:
            
    def __init__(self,dicom_basedir,ptcode,ctcode,basedir):
            
        import imageio
        #import easygui
        
        from stentseg.utils.datahandling import loadvol
        from stentseg.utils.datahandling import savecropvols, saveaveraged
        
        ## Select base directory for LOADING DICOM data
        
        #dicom_basedir = easygui.diropenbox()
        print('DICOM Path = ', dicom_basedir)
        
         
        #ctcode = '12months'  # 'pre', 'post_x', '12months'
        stenttype = 'nellix'      
        
        ## Select base directory to SAVE SSDF
        #basedir = easygui.diropenbox()
        print('Base Path = ', basedir)
        
        # Set which crops to save
        cropnames = ['prox'] #,'stent'] # ['ring'] or ['ring','stent'] or ..
        
        #===============================================================================
        ## Step A: read single volumes to get vols:
        #  folder1 = '10%'
        #  folder2 = '60%'
        #  vol1 = imageio.volread(os.path.join(dicom_basedir, folder1), 'dicom')
        #  vol2 = imageio.volread(os.path.join(dicom_basedir, folder2), 'dicom')
        #  print(  )
        #   
        #  if vol1.meta.SeriesDescription[:2] < vol2.meta.SeriesDescription[:2]:
        #      vols4078 = [vol1,vol2]
        #  else:
        #      vols4078 = [vol2,vol1]
        #       
        #  vols = vols4078.copy()
        #   
        #  for vol in vols:
        #      vol.meta.PatientName = ptcode # anonimyze
        #      vol.meta.PatientID = 'anonymous'
        #     print(vol.meta.SeriesDescription,'-', vol.meta.sampling)
        #===============================================================================
        
        ##Orginele code
        #===============================================================================
        # 
        # folder1 = '40% iDose'
        # folder2 = '78 iDose'
        # vol1 = imageio.volread(os.path.join(dicom_basedir, folder1), 'dicom')
        # vol2 = imageio.volread(os.path.join(dicom_basedir, folder2), 'dicom')
        # print(  )
        #  
        # if vol1.meta.SeriesDescription[:2] < vol2.meta.SeriesDescription[:2]:
        #     vols4078 = [vol1,vol2]
        # else:
        #     vols4078 = [vol2,vol1]
        #     
        # vols = vols4078.copy()
        # 
        # for vol in vols:
        #     vol.meta.PatientName = ptcode # anonimyze
        #     vol.meta.PatientID = 'anonymous'
        #     print(vol.meta.SeriesDescription,'-', vol.meta.sampling)
        #===============================================================================
        
        ## Step A: read 10 volumes to get vols
        # Deze zoekt alle mappen en dat zijn er dus 10 maar niet in de goede volgorde
        vols2 = [vol2 for vol2 in imageio.get_reader(dicom_basedir, 'DICOM', 'V')] 
        
        vols = [None] * len(vols2)
        for i, vol in enumerate(vols2):
        #    print(vol.meta.sampling)
            print(vol.meta.SeriesDescription)
            phase = int(vol.meta.SeriesDescription[:1]) 
            # use phase to fix order of phases
            vols[phase] = vol
            #vols[phase].meta.ImagePositionPatient = (0.0,0.0,0.0)
            
        for i,vol in enumerate(vols): #wat ik heb veranderd is i, en enumerate()
            print(vol.meta.SeriesDescription)
            assert vol.shape == vols[0].shape
            assert str(i*10) in vol.meta.SeriesDescription # 0% , 10% etc. 
        
        ## Step B: Crop and Save SSDF
        # 1 of 2 cropnames opgeven voor opslaan 1 of 2 crpos.
        # Het eerste volume wordt geladen in MIP, crop met marges van minimaal 30 mm
        for cropname in cropnames:
            savecropvols(vols, basedir, ptcode, ctcode, cropname, stenttype)
        #    saveaveraged(basedir, ptcode, ctcode, cropname, range(0,100,10))
        
        ## Visualize result
         
        #s1 = loadvol(basedir, ptcode, ctcode, cropnames[0], what ='10avgreg')
        #s2 = loadvol(basedir, ptcode, ctcode, cropnames[0], what ='10phases')
        s1 = loadvol(basedir, ptcode, ctcode, cropnames[0], what ='phases')
        #s2 = loadvol(basedir, ptcode, ctcode, cropnames[0], what = 'avg010')
        #vol1 = s1.vol
        vol1 = s1.vol40
         
        # Visualize and compare
        colormap = {'r': [(0.0, 0.0), (0.17727272, 1.0)],
         'g': [(0.0, 0.0), (0.27272728, 1.0)],
         'b': [(0.0, 0.0), (0.34545454, 1.0)],
         'a': [(0.0, 1.0), (1.0, 1.0)]}
          
        import visvis as vv
         
        fig = vv.figure(1); vv.clf()
        fig.position = 0, 22, 1366, 706
        a1 = vv.subplot(111)
        a1.daspect = 1, 1, -1
        # t1 = vv.volshow(vol1, clim=(0, 3000), renderStyle='iso') # iso or mip
        # t1.isoThreshold = 600 # stond op 400 maar je moet hoger zetten als je alleen stent wil
        # t1.colormap = colormap
        a1 = vv.volshow2(vol1, clim=(-500, 1500),renderStyle='mip')
        vv.xlabel('x'), vv.ylabel('y'), vv.zlabel('z')
        # vv.title('One volume at %i procent of cardiac cycle' % phase )
        vv.title('Vol40' )
         
