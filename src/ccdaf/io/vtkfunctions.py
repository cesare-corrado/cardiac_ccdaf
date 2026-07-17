import vtk
import os

def readvtk(filename : str) -> vtk.vtkPolyData:
    ''' readvtk(filename : str) -> vtk.vtkPolyData
    reads a vtk file and returns a vtkPolyData object
    '''
    path, extension = os.path.splitext(filename)
    extension       = extension.lower()
    reader          = None
    polydata        = None    
    if extension == ".ply":
        reader = vtk.vtkPLYReader()
    elif extension == ".vtp":
        reader = vtk.vtkXMLPolyDataReader()
    elif extension == ".obj":
        reader = vtk.vtkOBJReader()
    elif extension == ".stl":
        reader = vtk.vtkSTLReader()
    elif extension == ".vtk":
        # this modification should make this function working also for unstructured grids
        with open(filename,'rb') as ff:
            header = []
            for jj in range(4):
                header.append(ff.readline())
        poly_type = header[-1].strip().split()[-1].decode('ascii')
        if poly_type=='UNSTRUCTURED_GRID':
            print('reading data in unstructured grid format',flush=True)
            reader = vtk.vtkUnstructuredGridReader()
            reader.SetFileName(filename)
            reader.ReadAllScalarsOn()
            reader.ReadAllVectorsOn()
            reader.Update()

            geometry_filter = vtk.vtkGeometryFilter()
            geometry_filter.SetInputData(reader.GetOutput())
            geometry_filter.Update()
            polydata = geometry_filter.GetOutput()            
        else:    
            reader = vtk.vtkPolyDataReader()
    elif extension == ".g":
        reader = vtk.vtkBYUReader()
    if reader is not None and polydata is None:
        if extension == ".g":
            reader.SetGeometryFileName(filename)
        else:
            reader.SetFileName(filename)
        reader.ReadAllScalarsOn()
        reader.ReadAllVectorsOn()    
        reader.Update()
        polydata = reader.GetOutput()
    return polydata


def writevtk(polydata : vtk.vtkPolyData,filename : str,binary : bool = False):
    writer = vtk.vtkPolyDataWriter()
    if( (vtk.VTK_MAJOR_VERSION>9) or ( vtk.VTK_MAJOR_VERSION==9 and vtk.VTK_MINOR_VERSION >=2 ) ): 
            writer.SetFileVersion(42)
    writer.SetInputData(polydata)
    if(binary):
      writer.SetFileTypeToBinary()
    else:
      writer.SetFileTypeToASCII()
    writer.SetFileName(filename)
    writer.Write()




