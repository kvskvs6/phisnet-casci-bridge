import numpy as np
import h5py
import scipy.constants

def write_orbfile(filename,orbcoef_mat,index_mat):
    '''Write a Molcas (Ras)Orb file:
    
    filename: name of created file
    orbcoeff_mat: square array where collumns are orbital coefficients
    index_mat: array of 1 character strings defining active space
    
    Only case with no symmetry (1 irreducible representation)
    is implemented.
    '''
    
    assert len(orbcoef_mat.shape)==2
    assert len(set(orbcoef_mat.shape))==1
    assert orbcoef_mat.shape[0]==index_mat.shape[0]
    assert index_mat.dtype=='<U1' #assuming a little endian machine
    
    norb=orbcoef_mat.shape[0]
    
    with open(filename,'w') as stream:
        stream.write('#INPORB 2.0\n')
        stream.write('#INFO\n')
        stream.write('* Orbitals generated from predicted Fock matrix\n')
        stream.write('{:8}{:8}{:8}\n'.format(0,1,0)) #1 for one irreducible representation and second 0 for unknown origin
        stream.write('{0:8}\n{0:8}\n'.format(norb))
        
        stream.write('#ORB\n')
        for orb in range(norb):
            stream.write('* ORBITAL{:5}{:5}\n'.format(1,orb+1))
            for coef in range(norb):
                stream.write(' {: .14E}'.format(orbcoef_mat[coef,orb]))
                if (coef+1)%5==0 or coef==norb-1:
                    stream.write('\n')
        
        # Aparently need to add #OCC and #ONE sections even if not of interest to us
        stream.write('#OCC\n* OCCUPATION NUMBERS\n')
        for orb in range(norb):
            stream.write('  {:1.4f}'.format(1))
            if (orb+1)%10==0 or orb==norb-1:
                    stream.write('\n')
        
        stream.write('#ONE\n* ONE ELECTRON ENERGIES\n')
        for orb in range(norb):
            stream.write(' {: .4E}'.format(0))
            if (orb+1)%10==0 or orb==norb-1:
                    stream.write('\n')
                    
        stream.write('#INDEX\n')
        stream.write('* 1234567890\n')
        for orb in range(norb):
            if orb%10==0:
                stream.write(str(orb//10)[-1]+' ')
            stream.write(index_mat[orb])
            if (orb+1)%10==0:
                stream.write('\n')
        

def read_orbener(file):
    '''Read MO energies from a Molcas (Ras-)Orb file.'''
    
    energies=[]
    with open(file,'r') as stream:
        got_energies=False
        anchor="pre-start"
        while not got_energies and anchor!="":
            anchor=stream.readline()
            if "#INFO" in anchor and "orbitals" in stream.readline():
                stream.readline()
                norbitals=int(stream.readline())
                if (norbitals%10)!=0:
                    readlines=norbitals//10+1
                else:
                    readlines=norbitals//10
            elif "#ONE" in anchor and "ONE ELECTRON ENERGIES" in stream.readline():
                for i in range(readlines):
                    energies=energies+[float(e) for e in stream.readline().split()]
                got_energies=True
                
    return np.array(energies)
        

def read_orbcoef(file):
    '''Read MO coefficients from a Molcas (Ras)Orb file.'''
    
    orbitals=[]
    with open(file,"r") as stream:
        got_orbitals=False
        anchor="pre-start"
        while not got_orbitals and anchor!="":
            anchor=stream.readline()
            if "#INFO" in anchor and "orbitals" in stream.readline():
                stream.readline()
                norbitals=int(stream.readline())
                if (norbitals%5)!=0:
                    block=norbitals//5+1
                else:
                    block=norbitals//5
            elif "#ORB" in anchor:
                for orb in range(1,norbitals+1):
                    stream.readline()
                    coef_st=""
                    for row in range(block):
                        coef_st=coef_st+stream.readline()
                    orbitals.append(np.array(coef_st.split(),dtype="float"))
                got_orbitals=True
                
    return np.array(orbitals).T #C matrix should have MO coefficients as columns


def read_molcas_nm(filename,natoms):
    '''Read mass-weighted normal modes from Molcas log file.
    Molcas (and Gaussian) outpus the normal modes (eigen vectors of
    the mass-weighted Hessian) are scaled by the squaretoot of
    atomic masses in g.mol^-1.
    The from numerical or analytical frequency calculations may be different,
    specifically in excluding or including translational and rotational modes.
    This function tries to deal with both cases (only for non-linear molecules),
    but the output will exclude the 6 first listed modes corresponding to
    translations and rotations, if they are present.'''
    
    frames=(3*natoms-6)//6
    col_last_frame=(3*natoms-6)%6
    
    #array to store normal mode information
    nm_matrix=np.zeros((3*natoms,3*natoms-6),dtype=float)
    
    #tag signalling normal mode information in log file
    read_tag=False
      
    #tag to check if translational and rotational modes are included
    excTR_tag=False
    
    with open(filename,'r') as file:
        for line in file:
            if 'Note that rotational and translational degrees have been automatically removed' in line:
                excTR_tag=True
            
            elif 'Harmonic frequencies in cm-1' in line:
                read_tag=True
                
            elif read_tag and 'Red. mass:' in line:
                #skip one line
                next(file)
                
                #if TR modes not excluded, skip entire frame = 3*natoms+9 lines
                if not excTR_tag:
                    for _ in range(3*natoms+9):
                        next(file)
                
                #read full frames of 6
                for i in range(frames):
                    #read nm data
                    for j in range(3*natoms):
                        #conversion to float happens because of nm_matrix dtype
                        nm_matrix[j,i*6:(i+1)*6]=file.readline().split()[-6:]
                    #skip 9 lines
                    for j in range(9):
                        next(file)
                        
                if col_last_frame!=0:
                    #read last frame with less than 6 collumns
                    #read nm data
                    for j in range(3*natoms):
                        #conversion to float happens because of nm_matrix dtype
                        nm_matrix[j,-col_last_frame:]=file.readline().split()[-col_last_frame:]
                        
                #finished reading
                #got what we wanted so break off loop and close file
                break
                        
        return nm_matrix
    
    
def read_molcas_frequencies(filename,natoms):
    '''Read frequencies (in cm-1) from Molcas log file.
    The from numerical or analytical frequency calculations may be different,
    specifically in excluding or including translational and rotational modes.
    This function tries to deal with both cases (only for non-linear molecules),
    but the output will exclude the 6 first listed frequencies corresponding to
    translations and rotations, if they are present.'''
    
    #if TR modes are included there will be one more frame which is not read in
    frames=(3*natoms-6)//6
    col_last_frame=(3*natoms-6)%6 
    
    #array to store frequencies
    frequencies=np.zeros(3*natoms-6,dtype=float)
    
    #tag signalling normal mode information in log file
    read_tag=False
    
    #frames counter
    iframe=-1
    with open(filename,'r') as file:
        for line in file:
            if 'Note that rotational and translational degrees have been automatically removed' in line:
                #advance frame counter
                iframe=iframe+1
            
            elif 'Harmonic frequencies in cm-1' in line:
                read_tag=True
                
            elif read_tag and "Frequency:" in line:
                #if TR modes are included, skip first frame
                #(How does this work for transition states?)
                if iframe>=0 and iframe<frames:
                    frequencies[iframe*6:(iframe+1)*6]=line.split()[1:]
                elif iframe==frames:
                    if col_last_frame!=0:
                        frequencies[-col_last_frame:]=line.split()[1:]
                    
                    #finished reading, switch off read_tag
                    read_tag=False
                
                iframe=iframe+1
                
    return frequencies


def read_molcas_h5_hessian_masses(h5filename):
    '''Read information stored in h5 file produced by SLAPAF module when frequencies are
    calculated numerically (like for a state averaged CASSCF calculation).
    Output is a tuple with the raw Hessian and atomic masses, both in atomic units.
    Atomic masses are output in the same atom order used to compute the Hessian.'''
    
    #open h5 file
    file=h5py.File(h5filename,'r')
    
    dof=file.attrs['DOF']
    masses=file["CENTER_MASSES"][()] #stored in atomic units (ie electron mass)
    
    hessian=np.zeros((dof,dof),dtype=float)
    
    #only lower triangle of raw (non mass-weighted) Hessian
    #is stored in h5 file
    #populate hessian using lower triangluar indices
    hessian[np.tril_indices(dof)]=file["HESSIAN"]
    
    #close h5 file
    file.close()
    
    #rebuild upper triangle
    hessian=hessian+hessian.T
    
    #divide diagonal by 2
    hessian[np.diag_indices(dof)]=hessian[np.diag_indices(dof)]/2
    
    return (hessian,masses)


def read_molcas_h5_MWhessian(h5filename):
    '''Compute mass-weighted Hessian, from a h5 file produced by SLAPAF module when
    frequencies are calculated numerically (like for a state averaged CASSCF calculation).'''
    
    hessian,masses=read_molcas_h5_hessian_masses(h5filename)
    
    dof=3*len(masses)
    
    #i//3 is integer division and used to take same mass for 3 cartesian coordinates
    masses_mat=np.array([[masses[i//3]*masses[j//3] for i in range(dof)] for j in range(dof)])
    
    return hessian/(masses_mat)**0.5


def read_molcas_h5_xyz2nm_trans(h5filename):
    '''Read information of a numerical frequency calculation in a h5 file
    generated by SLAPAF.
    Output a transformation matrix D for Q=Dx, where
    x is a cartesian displacement in Angstrom
    Q is a frequency scalled normal mode displacement.
    Rows of D correspond to different normal modes, and columns to
    cartesians coordinates.'''
    
    hessian,masses=read_molcas_h5_hessian_masses(h5filename)
    
    dof=3*len(masses)
    
    #i//3 is integer division and used to take same mass for 3 cartesian coordinates
    masses_mat=np.array([[masses[i//3]*masses[j//3] for i in range(dof)] for j in range(dof)])
    
    MWhessian=hessian/(masses_mat)**0.5
    
    k,nm=np.linalg.eigh(MWhessian)
    
    #nm/(mass_i)**0.5 with atomic masses in g.mol^-1
    #is the value reported in the Molcas (and Gaussian) log files
    #no need to make that conversion here and use atomic units instead
    
    #discard 6 first values corresponding to translations and rotations
    #freq is in Hartrees
    freq=(k[6:])**0.5
    
    #ok to squareroot again
    scale=np.array([[(masses[j//3]*freq[i])**0.5 for i in range(dof-6)] for j in range(dof)])
    
    #include conversion from Angstrom to Bohr
    xyz2nm=((nm[:,6:]*scale).T)/(scipy.constants.physical_constants['Bohr radius'][0]*1e10)
    
    return xyz2nm


def read_molcas_h5_nm2xyz_trans(h5filename):
    '''Read information of a numerical frequency calculation in a h5 file
    generated by SLAPAF.
    Output a transformation matrix D' for x=D'Q, where
    x is a cartesian displacement in Angstrom
    Q is a frequency scalled normal mode displacement.
    Columns of D' correspond to different normal modes, and rows to
    cartesians coordinates.'''
    
    hessian,masses=read_molcas_h5_hessian_masses(h5filename)
    
    dof=3*len(masses)
    
    #i//3 is integer division and used to take same mass for 3 cartesian coordinates
    masses_mat=np.array([[masses[i//3]*masses[j//3] for i in range(dof)] for j in range(dof)])
    
    MWhessian=hessian/(masses_mat)**0.5
    
    k,nm=np.linalg.eigh(MWhessian)
    
    #nm/(mass_i)**0.5 with atomic masses in g.mol^-1
    #is the value reported in the Molcas (and Gaussian) logfiles
    #no need to make that conversion here and use atomic units instead
    
    #discard 6 first values corresponding to translations and rotations
    #freq is in Hartrees
    freq=(k[6:])**0.5
    
    #ok to squareroot again
    scale=np.array([[(masses[j//3]*freq[i])**0.5 for i in range(dof-6)] for j in range(dof)])
    
    #include conversion from Bohr to Angstrom
    nm2xyz=(nm[:,6:]/scale)*(scipy.constants.physical_constants['Bohr radius'][0]*1e10)
    
    return nm2xyz


def read_molcas_h5_overlap(h5filename):
    '''Read information of the overlap matrix in the atomic orbital basis
    from a h5 file generated by RASSCF.
    Output the square matrix and a numpy array.'''
    
    #open h5 file
    file=h5py.File(h5filename,'r')
    
    nbas=file.attrs['NBAS'][0]
    overlap=file['AO_OVERLAP_MATRIX'][()].reshape(nbas,nbas)
    
    #close h5 file
    file.close()
    
    return overlap