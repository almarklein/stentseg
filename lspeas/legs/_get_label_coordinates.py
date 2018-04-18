""" Script to get start/end points for a centerline calculation

"""

## point 1
n1coordinates = np.asarray(label2worldcoordinates(label1),dtype=np.float32) # x,y,z

## point 2

n2coordinates = np.asarray(label2worldcoordinates(label1),dtype=np.float32) # x,y,z




## point start

start1 = (n1coordinates+n2coordinates)/2
start1 = tuple(start1.flat)

## vis start point
view = a0.GetView()
p1 = vv.plot(start1[0], start1[1], start1[2], ms='.', ls='', mc='g', mw=15, axes=a0) # start1
a0.SetView(view)




## point end

end1 = (n1coordinates+n2coordinates)/2
end1 = tuple(end1.flat)

## vis end point

view = a0.GetView()
vv.plot(end1[0], end1[1], end1[2], ms='.', ls='', mc='r', mw=15, axes=a0) # start1
a0.SetView(view)

