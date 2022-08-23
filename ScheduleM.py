import numpy as np
from geneticalgorithm2 import geneticalgorithm2 as ga
import itertools as it

class Schedule:
            
    def __init__(self,C,T,S,R,days):
        self.C = C # Array of coefficients of importance by: NDates, NClassrooms, MinDist_Ex and MinSlack.
        self.T = T # Teacher - exams array
        self.S = S # Students - Exams array
        self.R = R # Capacity in every classroom array
        self.days = days # Number of days
        
        
        # Calculated objects ###########################################################
        ################################################################################
        
        #Parameters
        global n,x,m,w,z
        #Number of exams, classrooms, Students, Teachers, time-slots
        n,x,m,w,z = len(self.T[0]),len(self.R), len(self.S), len(self.T), days*4
        
        ################################################################################
        ################################################################################
        
        
    # Methods ######################################################################
    ################################################################################

    #Return: Number of examns by every time-slot (To teachers and students).
    def NEx_Ts(self, Ind_RP):
        NX = len(Ind_RP)
        NP_Ind = np.zeros((NX,z))

        for j in range(NX):
            for i in range(z):
                NP_Ind[j,i] = np.sum(Ind_RP[j,i*x:i*x+x])
        return NP_Ind

    # Return: The number of constraints that the solution violates.   
    def test(self,X,cond = True): # X as numpy array
        # Arrays
        global EPR,SRP,TRP,NSRP,NE_S_Ts,NE_T_Ts,ColSum_EPR
        
        # Exams - room*time-slots array
        EPR = np.reshape(X,(n,z*x)).copy()
        # Student - room*time-slots array
        SRP = np.dot(self.S,EPR)
        # Teacher - room*time-slots array
        TRP = np.dot(self.T,EPR)
        #Number of students by room at a time-slot
        NSRP = np.reshape(np.sum(SRP,axis=0),(z,x)).copy()
        #Number of exams of students by every time-slot
        NE_S_Ts = self.NEx_Ts(SRP)
        #Number of exams of teachers by every time-slot
        NE_T_Ts = self.NEx_Ts(TRP)
        # Occupied room by every time-slot
        ColSum_EPR = np.sum(EPR,axis=0)

        # To count the number of constraint violated 
        pen = np.zeros(5)

        #Every examn must be assign to a room SumRow_EPR = 1
        pen[0] = np.sum(np.where(np.sum(EPR,axis=1) != 1,1,0))

        #Every room cannot be booking for 2 examns SumRow_EPR <= 1
        pen[1] = np.sum(np.where(ColSum_EPR > 1,1,0))

        # Every student don´t must be in 2 exams at the same time
        pen[2] = np.sum(np.where(NE_S_Ts>1,1,0))

        # Every Teacher don´t must be in 2 exams at the same time
        pen[3] = np.sum(np.where(NE_T_Ts>1,1,0))

        # Number of students must be less than number of seats at the room
        for i in range(z):
            for j in range(x):
                if NSRP[i,j] > self.R[j]:
                    pen[4] += 1

        # It can return the number of constraints violated (1)
        # or an array of the number constraints violated for each one (2)
        if cond == True:
            return sum(pen)
        else:
            return pen
        
    # Return: The minimum distance between examns for every individual.
    def MinDist_Ex(self, NEx_ITs):
        #Taking index of every value different of zero
        Mat = []
        for j in range(np.shape(NEx_ITs)[0]):
            Mat.append([i for i in range(len(NEx_ITs[j,:])) if NEx_ITs[j,i] != 0])

        # Substacting every index, to determine the distance between every nonzero value
        # in the matrix of Number of exams - Individual*TimeSlot (Students o teachers)
        distance = []
        for i in range(len(Mat)):
            for j in range(len(Mat[i])-1):
                distance.append(Mat[i][j+1]-Mat[i][j])
        if distance != []:
            return np.min(distance)
        else:
            return -50000
        
    #Return: The number of occupated room.
    def NClassrooms(self,ColSum_EPR,cond = 1):
        #If the input data have the right shape (In GA)
        if cond == 1:
            return np.sum(np.where(np.sum(np.reshape(ColSum_EPR,(z,x)).copy(), axis = 0)!= 0,1,0))
        #If the input data don´t have the right shape (for output of GA)
        elif cond == 2:
            TimeSlots_Rooms = np.reshape(np.sum(ColSum_EPR,axis=0),(z,x)).copy()
            return np.sum(np.where(np.sum(TimeSlots_Rooms, axis = 0)!= 0,1,0))
        
    #Return: The min slack between the number of students 
    # in a exam and the capacity of the classroom.
    def MinSlack(self, NSRP, cond = 1):
        #If the input data have the right shape (In GA)
        if cond == 1:
            NStudents_InRoom = NSRP
        #It the input data dont have the right shape (Output GA)
        elif cond ==2:
            NStudents_InRoom = np.reshape(np.sum(np.dot(S,EPR),axis=0),(z,x))
        slacks = []
        for i in range(z):
            for j in range(x):
                if NStudents_InRoom[i,j] != 0:
                    slacks.append(self.R[j]- NStudents_InRoom[i,j])
        if slacks == [] or np.min(slacks) <= 0:
            return -50000
        else:
            return int(np.min(slacks))
    # Return: The number of days occupated in the schedule
    def NDates(self, ColSum_EPR,cond = 1):
        Occupated_Day = np.zeros((1,self.days))
        for i in range(self.days):
            #Number of exams by day
            if cond == 1: #In GA
                NEx_day = np.sum(ColSum_EPR[i*z:(i+1)*z])
            elif cond ==2: #Output GA
                NEx_day = np.sum(np.sum(ColSum_EPR,axis=0)[i*z:(i+1)*z])
            if NEx_day != 0:
                Occupated_Day[0,i] = 1
        return int(np.sum(Occupated_Day))
    
    ################################################################################
    #---------------------------- Genetic algorithm -------------------------------#
    ################################################################################


    def function(self,X,cond = True): # X as numpy array

        if cond == True:
            #Penalization #############################################
            pen = self.test(X)
            
            #Objective functions ######################################
            return self.C[0]*self.NDates(ColSum_EPR) + self.C[1]*self.NClassrooms(ColSum_EPR) - self.C[2]*self.MinDist_Ex(NE_S_Ts) - self.C[3]*self.MinSlack(NSRP) + pen**4*5000
        
        else:
            pen = self.test(X,False)
            return print(f'Restrictions violated: {pen} \n'+
                         f'Number of occupied dates: {self.NDates(ColSum_EPR)}\n'+
                         f'Number of occupied rooms: {self.NClassrooms(ColSum_EPR)}\n'+
                         f'Minimum distance between exams, for students: {self.MinDist_Ex(NE_S_Ts)}\n'+
                         f'Minimum slack for the occupied classrooms: {self.MinSlack(NSRP)}\n')
    
    # Run input parameters ##################################
    # no_gen : Max number of generations
    # pop_size : Number of individuals on the population
    # p_mut : Mutation probability
    # p_cross : crossover_probability
    # wo_improv = max number of iterations without improvement
    def run(self, no_gen = 100, pop_size = 100, p_mut = 0.1, p_cross = 0.5, wo_improv =  None):
        
        algorithm_param = {
                           'max_num_iteration': no_gen,
                           'population_size': pop_size,
                           'mutation_probability': p_mut,
                           'crossover_probability': p_cross,
                           'max_iteration_without_improv' : wo_improv
                           }
        
        model = ga(self.function, dimension = z*x*n, variable_type='bool',algorithm_parameters=algorithm_param)
        model.run(no_plot = True, disable_progress_bar = True, disable_printing = True)
        
        # To see the number of hard constraints violated
        model.output_dict['hardC_violated'] = self.test(model.output_dict['variable'])
        if self.test(model.output_dict['variable']) == 0:
            print(sc(model.output_dict['variable']))
        return model.output_dict
    
    def sc(self, X):
        M = np.where(np.reshape(X,(n,x*z)))
        for i in range(len(M[0])):
            print(f'Examen {M[0][i]+1} en el día {M[1][i]//(z)+1}, periodo {M[1][i]//2 % 4+1} y salón {M[1][i]%2+1}')
