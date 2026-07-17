import os
import numpy as np

###################################
###                             ###
### functions to read xml files ###
###                             ###
###################################

import xml.etree.ElementTree as ET

def extract_lists_for_loader(carto_input_directory: str) -> dict:
    """
    extract the files in the carto directory that can be either a study or a mapping
    returns a dict with:
       * 'studies':  list of study file names (.xml)
       * 'mappings': list of mapping file  names (.mesh)
    Note:
       1. mappings are taken from the .mesh files: no parsing needed, and the
          stem is already the map name
       2. [MapName]_Points_Export.xml only lists the per-point export files
          ([MapName]_P[ID]_Point_Export.xml), so both are skipped unparsed
    """
    map_names   = []
    study_names = []
    undefined   = []

    for filename in os.listdir(carto_input_directory):
        if filename.endswith(".mesh"):
            map_names.append(filename)
        if filename.endswith(".xml") and not (filename.endswith("_Point_Export.xml") or filename.endswith("_Points_Export.xml")):
            xmlfname = os.path.join(carto_input_directory,filename)
            try:
                root = ET.parse(xmlfname).getroot()
                if root.tag == "Study" and "name" in root.attrib:
                    study_names.append(filename)
            except Exception:
                # Invalid XML, parsing error, etc.
                undefined.append(filename)
    candidates = {'mappings':map_names, 'studies':study_names}
    return(candidates)


def extract_map_list_names(studyfname: str) -> list:
    '''Given the study name (.xml file), it extracts the list of the associated mappings'''
    Maps_list_names = []
    root      = ET.parse(studyfname).getroot()
    Maps      = root.find('Maps')
    Maps_list = Maps.findall('Map') 
    Maps_list_names = []
    for MM in Maps_list:
        MName = MM.get('Name')
        Maps_list_names.append(MName)
    return(Maps_list_names)


###################################################
###                                             ###
### functions to mapping-related carto files    ###
###                                             ###
###################################################

def read_carto_mesh_file(mesh_filename: str) -> dict:
    """ 
    Extracts the data from the .mesh file and returns a python dictionary:
        * Nodes (X)
        * Triangles (Tri)
        * variables associated to the vertivces (VertexColors)
        * ColorsIDs, ColorsNames: the column indices (ColorsIDs) and the corresponding names (ColorsNames)
    Note: 
       1. vertexColors are interpolated by carto from data in _car file
       2. A value of -10000 flags invalid coloring data at a specific point       
       3. we DID NOT extract the section VerticesAttributesSection
    """
    
    mesh0 = {'X': None, 
             'Tri': None,
             'VertexColors': None}
    
    section_names = ['[GeneralAttributes]',
                     '[VerticesSection]','[TrianglesSection]',
                     '[VerticesColorsSection]',
                     '[VerticesAttributesSection]' ]
                     
    with open(mesh_filename,'rb') as fm:
        data   = fm.readlines()
        data   = data[4:]
        for row,rowdata in enumerate(data):
            data[row]=rowdata.strip()
            try:
                data[row]=data[row].decode('utf8')
            except Exception:
                data[row]=data[row].decode('latin-1')
    section_index_start = []
    for sname in section_names:
        index = data.index(sname) if sname in data else -1
        section_index_start.append(index)
    section_index_start      = np.array(section_index_start)
    nv = 0
    nT = 0
    nC = 0
    #General attributes
    I0 = section_index_start[0]
    I1 = section_index_start[section_index_start>I0].min()
    Attributes = []
    for jind in range(1+I0,I1):
        row = data[jind].strip().split('=')
        if row[0].strip()=='NumVertex':
            nv = int(row[1])
            mesh0['X'] = np.zeros((nv,3),dtype=float)
        elif row[0].strip()=='NumTriangle':
            nT = int(row[1])
            mesh0['Tri'] =  np.zeros((nT,3),dtype=int)
        elif row[0].strip()=='NumVertexColors':
            nC = int(row[1])
            mesh0['VertexColors'] = np.zeros((nv,nC),dtype=float)
        elif row[0].strip()=='ColorsIDs':
            key  = row[0].strip()
            vals = np.array(row[1].strip().split(),dtype=int)
            mesh0[key] = vals
            #Attributes.append([key,vals])
        elif row[0].strip()=='ColorsNames':
            key  = row[0].strip()
            vals = np.array(row[1].strip().split())
            mesh0[key] = vals
            #Attributes.append([key,vals])
        elif len(row[0])>0:
            if len(row)>1:
                Attributes.append([row[0].strip(),row[1].strip()])
    #VerticesSection attributes
    I0 = section_index_start[1]
    for jentry in range(nv):
        jind = 3+I0+jentry
        #ID = X,Y,Z,nx,ny,nz,groupID
        row = data[jind].strip().split()[2:5]
        mesh0['X'][jentry,:]=np.array(row,dtype=float)
    #TrianglesSection attributes
    I0 = section_index_start[2]
    for jentry in range(nT):
        jind = 3+I0+jentry
        #ID = v0,v1,v2,nx,ny,nz,groupID
        row = data[jind].strip().split()[2:5]
        mesh0['Tri'][jentry,:] = np.array(row,dtype=int)
    #VerticesColorsSection attributes
    I0 = section_index_start[3]
    if I0>=0 and nC>0:
        for jentry in range(nv):
            jind = 4+I0+jentry
            row = data[jind].strip().split()[2:]
            mesh0['VertexColors'][jentry,:]=np.array(row,dtype=float)
    return(mesh0)

   
def load_carto_electrodes(carto_elec: str)  -> dict: 
    """ 
    Reads the _car.txt file and extacts some useful quantities (in this order):
        * node id
        * [x,y,z] electrode coordinates
        * unipolar voltage
        * bipolar voltage
        * Local Activation Time
        * catheter id (internal identification)
    returns a dict with : 
        * 'Names': the name of each filed (columns of data)
        * 'data':  an np.ndarray (nelectrodes, nfields) of the data
    Note:
        1. Carto creates this file as "postprocessing" of all point data
        2. It summarises the quantrities provided by .mesh at electrode coordiantes
    """
    out_list_names = ['node_id','x_coord','y_coord','z_coord','unipolar_voltage','bipolar_voltage','LAT','cathID']

    import string
    import itertools
    def generate_excel_labels(n: int) -> list:
        """ 
        This is a helper function that generates a list of "columns" with the same labels used in 
        Excel. E.g. A to Z, then AA,...
        """
        labels = []
        base = string.ascii_uppercase  # 'A' to 'Z'
        # Generate labels iteratively
        for length in range(1, 3):  # Extend range as needed (1 = 'A'-'Z', 2 = 'AA'-'ZZ', etc.)
            for item in itertools.product(base, repeat=length):
                labels.append("".join(item))
                if len(labels) == n:
                    return labels
        return labels
        
    nb_of_letters = len(string.ascii_uppercase)
    # max column is AT: find the index of T to repeat from AA
    T_index       = list(string.ascii_uppercase).index('T')
    xc_lab        = generate_excel_labels(nb_of_letters+T_index+1)
    # Now I can use table 5 from carto manual
    Icolumns = np.array([ xc_lab.index('C'),       # Node id
                 xc_lab.index('E'),xc_lab.index('F'),xc_lab.index('G'),  #coords
                 xc_lab.index('K'),xc_lab.index('L'),xc_lab.index('M'), #voltage and LAT
                 xc_lab.index('T')    # This is the catheter ID; not sure if it is useful
                 ],dtype=int)
    with open(carto_elec,'r') as fcar:
        data_car = fcar.readlines()
        data_car = data_car[1:]
    if len(data_car)>0:
        for irow, row_values in enumerate(data_car):
            data_car[irow] = row_values.strip().split()
        data_car = np.array(data_car)[:,Icolumns]
        data_car = data_car.astype(float)
    else:
        # header-only file: a map with no acquired point
        data_car = np.zeros((0,len(Icolumns)),dtype=float)
    electrodes  = {'Names': out_list_names, 'data': data_car }
    return electrodes

    
    
    
    
