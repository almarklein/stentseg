"""

===========================================
conda install vispy
===========================================

"""
try:
    from PyQt5 import QtCore, QtGui # PyQt5
    from PyQt5.QtWidgets import QDialog as QDialog
    from PyQt5.QtWidgets import *
except ImportError:
    from PySide import QtCore, QtGui # PySide2
    from PySide.QtGui import QDialog as QDialog
    from PySide.QtGui import *

class MyDialog(QDialog): # QtGui.QDialog): 
    """ Pop-up dialog
    """
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Enter name of node comparison:')
        self.resize(300, 100 )
        
        # Create widgets
        self.edit = QLineEdit(self)
        self.button = QPushButton('O.K.', self)
        
        # Define interaction
        self.button.clicked.connect(self.close)
        layout = QVBoxLayout(self)

        # Put widgets in layout
        self.setLayout(layout)
        
        # layout.addWidget(self.text)
        layout.addWidget(self.edit)
        layout.addWidget(self.button)
        

# app = QtGui.QApplication([])
# m = MyDialog()
# m.show()
# m.exec_()
# nr_of_stents = int(m.edit.text())