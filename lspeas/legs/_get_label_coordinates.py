""" Script to get start/end points for a centerline calculation

"""

## point 1
n1coordinates = np.asarray(label2worldcoordinates(label1),dtype=np.float32) # x,y,z

## point 2

n2coordinates = np.asarray(label2worldcoordinates(label1),dtype=np.float32) # x,y,z

## midpoint

start1 = (n1coordinates+n2coordinates)/2
start1 = tuple(start1.flat)

## vis midpoint

vv.plot(start1[0], start1[1], start1[2], ms='.', ls='', mc='g', mw=15, axes=a0) # start1