index_of_ith_3mer_count_in_vec32 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 7, 11, 12, 13, 3, 14, 15, 16, 13, 17, 18, 19, 10, 20, 21,
                                    19, 6, 22, 23, 16, 2, 24, 25, 23, 12, 26, 27, 21, 9, 28, 27, 18, 5, 29, 25, 15, 1, 30, 29,
                                    22, 11, 31, 28, 20, 8, 31, 26, 17, 4, 30, 24, 14, 0]

vec32_index_i_could_be_3mer = [[0, 63], [1, 47], [2, 31], [3, 15], [4, 59], [5, 43], [6, 27], [7, 11], [8, 55], [9, 39], [10, 23],
                               [12, 51], [13, 35], [14, 19], [16, 62], [17, 46], [18, 30], [20, 58], [21, 42], [22, 26], [24, 54],
                               [25, 38], [28, 50], [29, 34], [32, 61], [33, 45], [36, 57], [37, 41], [40, 53], [44, 49], [48, 60],
                               [52, 56]]


# below script generated vec32_ith_could_be_3mer from pos_of_ith_3mer_in_vec32
def build_that_2nd_list():
    for i in range(64):
        l = index_of_ith_3mer_count_in_vec32
        u = vec32_index_i_could_be_3mer
        if l[i] in u.keys():
            u[l[i]].append(i)
        else:
            u[l[i]] = []
            u[l[i]].append(i)
## end

# get 3mer. ex: 0 -> AAA, 63 -> TTT
def get_3mer_given_its_vec64_index(vec64_index):
    k = 3
    mask = 3
    ch = [0,0,0]
    for i in range(3):
        comp = (mask & vec64_index) >> 2 * i
        if     comp == 0:
            ch[k-1-i] = 'A'
        elif comp == 1:
            ch[k-1-i] = 'C'
        elif comp == 2 :
            ch[k-1-i] = 'G'
        elif(comp == 3):ch[k-1-i] = 'T'

        mask = mask <<2
    return ch


# turn vec32_ith_could_be_3mer into human readable
def display_kmer_possibilities_given_a_vec32(k):
    kmers = []
    for i in range(32):
        for n in range(k[i]):kmers.append([get_3mer_given_its_vec64_index(vec32_index_i_could_be_3mer[i][0]),
                                        get_3mer_given_its_vec64_index(vec32_index_i_could_be_3mer[i][1])
                                        ])
    return kmers
