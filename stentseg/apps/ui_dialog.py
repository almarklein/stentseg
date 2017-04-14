"""

===========================================
conda install vispy
===========================================

"""
from PyQt4 import QtCore, QtGui

class MyDialog(QtGui.QDialog):
    """ Pop-up dialog
    """
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Enter name of node comparison:')
        self.resize(300, 100 )
        
        # Create widgets
        self.edit = QtGui.QLineEdit(self)
        self.button = QtGui.QPushButton('O.K.', self)
        
        # Define interaction
        self.button.clicked.connect(self.close)
        layout = QtGui.QVBoxLayout(self)

        # Put widgets in layout
        self.setLayout(layout)
        
        # layout.addWidget(self.text)
        layout.addWidget(self.edit)
        layout.addWidget(self.button)
        

# app = QtGui.QApplication([])
# m = MyDialog()
# m.show()
# m.exec_()
# nr_of_stents = int(m.edit.currentText())