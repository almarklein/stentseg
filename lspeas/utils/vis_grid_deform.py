from stentseg.utils.datahandling import select_dir

def imagePooper(reg, fname='test', checkSize=25):
    """
    """
    ims = reg._ims.copy()
    im1, im2 = ims[0], ims[1]
    im3 = reg.get_final_deform().apply_deformation(im1) # todo: this is incorrect for groupwise?
        # reg.get_final_deform(0,1).apply_deformation(im1) # from i to j?
    # Correct images  
    im1 -= im1.min()        
    im1 /= im1.max()
    im2 -= im2.min()        
    im2 /= im2.max()
    im3 -= im3.min()        
    im3 /= im3.max()
    # Make checker
    checker = np.zeros((im1.shape[0], im1.shape[1],3), dtype=np.float32)
    # Go
    y=0
    yWhich = 0
    while y<im1.shape[0]:
        x=0
        if yWhich:
            xWhich = 0
            yWhich = 0
        else:
            xWhich = 1
            yWhich = 1
        while x<im1.shape[1]:
            y2, x2 = y+checkSize, x+checkSize
            if xWhich:
                checker[y:y2,x:x2,0] = im3[y:y2,x:x2]
                checker[y:y2,x:x2,1] = im3[y:y2,x:x2]
                xWhich = 0
            else:
                checker[y:y2,x:x2,1] = im2[y:y2,x:x2]
                checker[y:y2,x:x2,2] = im2[y:y2,x:x2]
                xWhich = 1
            # Next
            x += checkSize
        y += checkSize
    # Show grid image
    grid1 = np.zeros(im1.shape, dtype=np.float32)
    size = 10
    grid1[:,::size] = 1
    grid1[:,1::size] = 1
    grid1[::size,:] = 1
    grid1[1::size,:] = 1
    grid2 = reg.get_final_deform().apply_deformation(grid1)
    # Save
    # path = 'C:/almar/report/_paper 2011 SPIE MI/images/'
    path = select_dir(r'D:\Profiles\koenradesma\Desktop',
                       r'C:\Users\Maaike\Desktop')
    dirim = path+'\\imagepooper'
    if not os.path.exists(dirim):
        os.makedirs(dirim)
    vv.imwrite(os.path.join(dirim,fname+'_1.png'), im1)
    vv.imwrite(os.path.join(dirim,fname+'_2.png'), im2)
    vv.imwrite(os.path.join(dirim,fname+'_r.png'), checker)
    vv.imwrite(os.path.join(dirim,fname+'_g.png'), grid2)
    print('images stored')
    
    return im1, im2, checker, grid2
    

if __name__ == '__main__':
    
    im1, im2, checker, grid = imagePooper(reg)
