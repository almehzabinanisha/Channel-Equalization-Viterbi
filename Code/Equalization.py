import itertools
import numpy as np
import scipy.stats
n,l = 0,0
w = []
sigma = 0
strTrainI,strTestI = [],[]
states,train_states,test_states,test_pred_states = None,[],[],[]
states_list = []
train_x,test_x = [],[]

class Transition:
    def __init__(self,from_,to_,prob):
        self.from_ = from_
        self.to_ = to_
        self.prob = prob


class States:
    def __init__(self,s,I,sigma):
        self.s = s
        self.I = I
        self.sigma = sigma
        #to be calculated later
        self.prior = 0
        self.transitions = []
        self.x = []
        self.miu = 0


def read_data():
    global n,l,w,sigma,strTrainI,strTestI
    print('Enter config.txt')
    path1 = input()
    file1 = open(path1,'r')
    line_count = 0
    for line in file1:
        newline = line[:-1]
        line1 = newline.split(" ")
        if line_count == 0:
            n = int(line1[0])
            l = int(line1[1])
            #print(n,l)
        elif line_count == 1:
            for e in line1:
                w.append(float(e))
            #print(w)
        else:
            sigma = float(line1[0])
            #print(sigma)
        line_count+=1
    print('Enter train.txt')
    path2 = input()
    file2 = open(path2,'r')
    line_count = 0
    for line in file2:
        line1 = line[:-1]
        if line_count ==0:
            strTrainI = list(line1)
        line_count += 1
    print('Enter test.txt')
    path3 = input()
    file3 = open(path3,'r')
    line_count = 0
    for line in file3:
        line1 = line[:-1]
        if line_count ==0:
            strTestI = list(line1)
        line_count += 1


def construct_states():
    global n,states,train_states,test_states,strTrainI,strTestI,states_list,sigma
    print('Construct states ')
    states = [''.join(x) for x in itertools.product('01', repeat=n)]
    total_states = len(states)
    for i in range(total_states):
        s = states[i]
        I = s[-1]
        st = States(s,I,sigma)
        print(s,I)
        states_list.append(st)
    #print(states)
    #print(strTrainI)
    #print(len(states_list))
    for i in range(len(strTrainI)):
        if i < (n-1):
            list1 = [''.join(x) for x in itertools.product('0', repeat=n-1-i)]
            #print(list1)
            string = ""
            for j in range(len(list1)):
                string += list1[j]
            for j in range(i+1):
                string += strTrainI[j]
            #print(string)
        else:
            string = ""
            for j in range(i-n+1,i+1,1):
                string += strTrainI[j]
            #print(string)
        train_states.append(string)
    #print(train_states)
    for i in range(len(strTestI)):
        if i < (n-1):
            list1 = [''.join(x) for x in itertools.product('0', repeat=n-1-i)]
            #print(list1)
            string = ""
            for j in range(len(list1)):
                string += list1[j]
            for j in range(i+1):
                string += strTestI[j]
            #print(string)
        else:
            string = ""
            for j in range(i-n+1,i+1,1):
                string += strTestI[j]
            #print(string)
        test_states.append(string)
    #print(test_states)


def calculate_prior_prob():
    print('calculate prior probability')
    global states_list,train_states,states
    for ls in states_list:
        c = train_states.count(ls.s)
        ls.prior = c / len(train_states)
        print(ls.s," ---> ",ls.prior)


def calculate_transition_prob():
    print('calculate transition probability')
    global states_list, train_states, states
    for ls in states_list:
        #get all the occurances of a cluster pattern
        indices = [i for i, x in enumerate(train_states) if x == ls.s]
        src = ls.s
        #get how many times that pattern appear in the training patterns
        #denom = train_states.count(src)
        #print(indices)
        idx_list = []
        for idx in indices:
            if idx < (len(train_states)-1): #if not the last index
                d = train_states[idx+1]
                idx_list.append(d) # append the next occuring pattern
        #print(idx_list)
        idx_set = list(set(idx_list))
        denom = len(idx_list)
        for v in states:
            if v in idx_set: #from the set of occuring pattern
                dest = v
                numer = idx_list.count(dest)
                prob = numer/denom
                trans = Transition(src,dest,prob)
                print("From ",src," ----> To ",dest," chances ",prob)
                ls.transitions.append(trans) #append that destination pattern
            else:
                dest = v
                prob = 0
                trans = Transition(src, dest, prob)
                print("From ", src," ----> To ", dest, " chances ", prob)
                ls.transitions.append(trans)  # append that destination pattern
        #print(len(ls.transitions))

def calculate_mean():
    print('mean calculation')
    global train_states,test_states,states_list,states,w,sigma
    global train_x,test_x
    w.reverse()
    #print(w)
    np.random.seed(0)
    for item in train_states:
        I_k = []
        for i in range(len(item)): #coz each pattern is a string
            I_k.append(int(item[i]))
        #print(I_k)
        #all string in the patterns are int format now
        cls = item[-1] #this is the class it belongs
        value1 = np.dot(np.reshape(I_k,(1,len(I_k))),np.reshape(w,(len(w),1)))[0][0]
        value2 = np.random.normal(0,sigma)
        #print(value1,value2)
        x = value1 + value2
        #print(value1, " + ", value2, " = ",x)
        train_x.append(x)
        #print(x)
    for item in test_states:
        I_k = []
        for i in range(len(item)): #coz each pattern is a string
            I_k.append(int(item[i]))
        #print(I_k)
        #all string in the patterns are int format now
        #cls = item[-1] #this is the class it belongs
        value1 = np.dot(np.reshape(I_k,(1,len(I_k))),np.reshape(w,(len(w),1)))[0][0]
        value2 = np.random.normal(0,sigma)
        #print(value1,value2)
        x = value1 + value2
        #print(value1, " + ", value2, " = ",x)
        test_x.append(x)
        #print(x)
    #print('After')
    for i in range(len(train_x)):
        st = train_states[i]
        x = train_x[i]
        lst = list(filter(lambda cls: cls.s == st, states_list))
        finlist = lst[0]
        #print(finlist.s,finlist.I)
        finlist.x.append(x)
    for obj in states_list:
        obj.miu = sum(obj.x) / len(obj.x)
        print(obj.s," mean ",obj.miu)


def calculate_viterbi():
    print('Viterbi Algorithm')
    global test_x,states_list,test_states,test_pred_states,sigma
    DP = [-9999 for j in range(len(test_x))]
    Clusters = [ "" for j in range(len(test_x))]
    test_pred_states = [ '0' for j in range(len(test_x))]
    #print(test_states)
    for i in range(len(test_x)):
        xval = test_x[i]
        #print(xval)
        if i == 0:
            for ob in states_list:
                cluster = ob.s
                cls = ob.I
                prior = ob.prior
                miu = ob.miu
                pdf = scipy.stats.norm.pdf(xval, miu, sigma)
                val = prior * pdf
                if val > DP[i]:
                    DP[i] = val
                    Clusters[i] = cluster
                    test_pred_states[i] = cls
                    print("iteration ",i," cluster ",Clusters[i]," class ",test_pred_states[i])
        else:
            clus = Clusters[i-1]
            lst = list(filter(lambda cls: cls.s == clus, states_list))
            finlist = lst[0]
            #print("From Object is ",finlist.s)
            t_list = finlist.transitions
            #print(len(t_list))
            for t in t_list:
                cluster = t.to_
                trans = t.prob #transition probability
                clst = list(filter(lambda cls: cls.s == cluster, states_list))
                fclst = clst[0]
                #print("To Object is ", fclst.s)
                cls = fclst.I
                miu = fclst.miu
                pdf = scipy.stats.norm.pdf(xval,miu,sigma)
                #print(xval,miu,pdf)
                val = trans * pdf * DP[i-1]
                if val > DP[i]:
                    DP[i] = val
                    Clusters[i] = cluster
                    test_pred_states[i] = cls
                    print("iteration ", i, " cluster ", Clusters[i], " class ", test_pred_states[i])
    print(test_pred_states)


def calculate_mismatch():
    print('calculate mismatch')
    total_count,count = 0,0
    print(strTestI)
    for i in range(len(strTestI)):
        total_count += 1
        if strTestI[i] != test_pred_states[i]:
            count += 1
            print("Position : ",i," Given : ",strTestI[i]," Predicted: ",test_pred_states[i])
    print(count)
    accuracy = (total_count - count)/total_count * 100
    print(accuracy)


if __name__ == '__main__':
    read_data()
    construct_states()
    calculate_prior_prob()
    calculate_transition_prob()
    calculate_mean()
    calculate_viterbi()
    calculate_mismatch()