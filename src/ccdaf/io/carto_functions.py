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

    
    
    
    


###################################################
###                                             ###
### functions to read point / EGM carto files   ###
###                                             ###
###################################################

# A Carto map stores, for every acquired point, its own
# [MapName]_P[ID]_Point_Export.xml. Points recorded together (one catheter
# placement) share one [MapName]_ECG_Export_[StartTime].txt, and every
# catheter connected at that moment has one
# [MapName]_[Connector]_Eleclectrode_Positions_OnAnnotation_[StartTime].txt
# ("Eleclectrode" is Carto's own spelling).

def get_exported_points(exported_pts_fname: str) -> list:
    ''' This function reads in the file <Mapping_name>_Points_Export.xml,
    extracts the list of the exported points
    and for each point it provides:
    * the point ID
    * the associated xml file
    example of a record from the file it is going to read:
       <Point ID="1" File_Name="Mapping_name_P1_Point_Export.xml" />
    '''
    PointList = []
    root      = ET.parse(exported_pts_fname).getroot()
    for pts in root.findall('Point'):
        PointList.append([pts.get("ID"), pts.get("File_Name")])
    return(PointList)


def get_point_info(pFName: str, studyfolder: str) -> dict:
    ''' This function reads a specific .xml file (example of file name is:
    Mapping_name_P1_Point_Export.xml) that is assigned to a point and provides:
        * 'ECG':         the name of: the ecg file, the Unipolar Mapping channel,
                         the Bipolar Mapping channel and the Reference_channel
        * 'StartTime':   the acquisition start time; it is also the number in the
                         name of the ECG and position files of this point
        * 'Annotations': [reference annotation, map annotation], as sample indices
                         of the ECG file
        * 'Voltages':    [unipolar, bipolar] voltage
        * 'Connectors':  a list with the names of the connectors recorded for that point
        * 'CycleLength': the cycle length computed by carto from the reference
                         annotation (None when the export has no ReferenceAnnotations)
        * 'WOI':         [From, To] window of interest, relative to the reference annotation
        * 'Impedances':  [Rate, Number]

    Note: A point corresponds to e.g. an electrode at a specific location.
          When the catheter is moved to a different location, the electrode
          will generate a new point.
    '''
    root = ET.parse(os.path.join(studyfolder, pFName)).getroot()
    ecg  = root.find("ECG")
    # ECG file and channel names
    ECGFname                 = ecg.get('FileName')
    Unipolar_Mapping_channel = ecg.get('UnipolarMappingChannel')
    Bipolar_Mapping_channel  = ecg.get('BipolarMappingChannel')
    Reference_channel        = ecg.get('ReferenceChannel')
    # Annotations
    annotations    = root.find("Annotations")
    start_time     = annotations.get('StartTime')
    start_time     = int(start_time) if start_time is not None else None
    ref_annotation = int(annotations.get('Reference_Annotation'))
    map_annotation = int(annotations.get('Map_Annotation'))
    # Voltages
    unipolar_volt  = float(root.find("Voltages").get("Unipolar"))
    bipolar_volt   = float(root.find("Voltages").get("Bipolar"))
    # Connector names
    connector_names = set()
    Positions = root.find('Positions')
    if Positions is not None:
        for jcc in Positions.findall("Connector"):
            connector_names.add(list(jcc.attrib.keys())[0])
    connector_names = sorted(connector_names)

    # WOI
    WOI  = root.find("WOI")
    From = None
    To   = None
    if WOI is not None:
        From = WOI.get('From')
        From = int(From) if From is not None else None
        To   = WOI.get('To')
        To   = int(To)   if To is not None else None

    # Impedances
    Impedances = root.find('Impedances')
    Rate   = None
    Number = None
    if Impedances is not None:
        Rate   = Impedances.get('Rate')
        Rate   = int(Rate) if Rate is not None else None
        Number = Impedances.get('Number')
        Number = int(Number) if Number is not None else None

    # Cycle lengths
    ref = root.find('ReferenceAnnotations')
    cycle_length = ref.get('CycleLength') if ref is not None else None
    cycle_length = int(cycle_length) if cycle_length is not None else None

    return({ 'ECG': [ECGFname, Unipolar_Mapping_channel, Bipolar_Mapping_channel, Reference_channel],
             'StartTime': start_time,
             'Annotations': [ref_annotation, map_annotation],
             'Voltages': [unipolar_volt, bipolar_volt],
             'Connectors': connector_names,
             'CycleLength': cycle_length,
             'WOI': [From, To],
             'Impedances': [Rate, Number] })


# Fill value for a trace that is not in a given ECG file. Raw counts seen so
# far stay within about +/-1100, far from the int16 limit, so the most
# negative int16 can never be a real sample.
MISSING_TRACE = np.iinfo(np.int16).min


def _ecg_channel_name(header_token: str) -> str:
    """ 'CS1-CS2(101)' -> 'CS1-CS2': the header appends the channel number in brackets """
    k = header_token.rfind('(')
    return header_token[:k] if k > 0 and header_token.endswith(')') else header_token


def read_ecg_header(ECGFname: str) -> tuple:
    ''' Reads only the header of an ECG export file and returns
    (channel names, gain), where the channel names have the "(n)" suffix removed
    and gain converts the raw counts to mV.
    The file layout is:
        line 1: format (e.g. ECG_Export_4.1)
        line 2: Raw ECG to MV (gain) = 0.003000
        line 3: the channel names
        then one row per sample (1 kHz), one column per channel, raw integer counts
    '''
    with open(ECGFname, 'r') as fecg:
        fecg.readline()
        gain  = float(fecg.readline().strip().split('=')[1])
        names = [_ecg_channel_name(tok) for tok in fecg.readline().split()]
    return names, gain


def read_ecg_file(ECGFname: str, channels: list) -> tuple:
    ''' Reads the EGM file with name ECGFname and returns the raw traces of the
    requested channels (2.5 s at 1 kHz in the exports seen so far):
        * traces:  np.int16 array (len(channels), n_samples) of raw counts;
                   multiply by gain to get mV. A channel missing from this file
                   is filled with MISSING_TRACE.
        * gain:    float, mV per count
        * present: bool array (len(channels),), False for a missing channel
    Channels are matched by exact name (the header "(n)" suffix removed), so
    e.g. '20A_1' never matches '20A_10' or '20A_1-2'.
    '''
    with open(ECGFname, 'r') as fecg:
        lines = fecg.read().split('\n', 3)
    gain  = float(lines[1].strip().split('=')[1])
    names = [_ecg_channel_name(tok) for tok in lines[2].split()]
    data  = np.array(lines[3].split(), dtype=np.int64)
    if data.size % len(names):
        raise ValueError(f"{os.path.basename(ECGFname)}: {data.size} values do not fill "
                         f"{len(names)} channels")
    data  = data.reshape(-1, len(names))
    if data.size and (data.min() <= MISSING_TRACE or data.max() > np.iinfo(np.int16).max):
        raise ValueError(f"{os.path.basename(ECGFname)}: raw values outside the int16 range")
    index   = {n: i for i, n in enumerate(names)}
    present = np.array([c in index for c in channels], dtype=bool)
    traces  = np.full((len(channels), data.shape[0]), MISSING_TRACE, dtype=np.int16)
    for j, c in enumerate(channels):
        if present[j]:
            traces[j] = data[:, index[c]]
    return traces, gain, present


def read_electrode_positions(pos_fname: str) -> tuple:
    ''' Reads a [MapName]_[Connector]_Eleclectrode_Positions_OnAnnotation_[StartTime].txt
    file: the position of every electrode of one catheter at the annotation time.
    Returns (electrode ids, xyz):
        * electrode ids: int array (n,), the 1st column; it restarts from 1 on
                         every spline of a multi-spline catheter
        * xyz:           float array (n, 3)
    The first two lines are a header (format version, column names).
    '''
    ids = []
    xyz = []
    with open(pos_fname, 'r') as fpos:
        for row in fpos.read().splitlines()[2:]:
            tok = row.split()
            if len(tok) >= 5:
                ids.append(int(tok[0]))
                xyz.append([float(v) for v in tok[2:5]])
    return np.array(ids, dtype=int), np.array(xyz, dtype=float).reshape(-1, 3)
