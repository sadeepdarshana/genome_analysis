# from matplotlib import pyplot as plt
# from celluloid import Camera
# import numpy as np
#
#
# # create figure object
# fig = plt.figure()
# # load axis box
#
# camera = Camera(fig)
# for i in range(10):
#     plt.scatter(i, np.random.random())
#     plt.scatter(i, np.random.random())
#     plt.scatter(i, np.random.random())
#     plt.scatter(i, np.random.random())
#     camera.snap()
#     plt.scatter(i, np.random.random())
#     plt.scatter(i, np.random.random())
#     plt.scatter(i, np.random.random())
#     plt.show()
# camera.snap()
# plt.show()
# animation = camera.animate()
# animation.save('test/animation.gif', writer='PillowWriter', fps=2)

tetramers = ['AAAA','AAAC','AAAG','AAAT','AACA','AACC','AACG','AACT','AAGA','AAGC','AAGG','AAGT','AATA','AATC', 'AATG','AATT','ACAA','ACAC','ACAG','ACAT','ACCA','ACCC','ACCG','ACCT','ACGA','ACGC','ACGG','ACGT','ACTA','ACTC','ACTG','AGAA','AGAC','AGAG','AGAT','AGCA','AGCC','AGCG','AGCT','AGGA','AGGC','AGGG','AGTA','AGTC','AGTG','ATAA','ATAC','ATAG','ATAT','ATCA','ATCC','ATCG','ATGA','ATGC','ATGG','ATTA','ATTC','ATTG','CAAA','CAAC','CAAG','CACA','CACC','CACG','CAGA','CAGC','CAGG','CATA','CATC','CATG','CCAA','CCAC','CCAG','CCCA','CCCC','CCCG','CCGA','CCGC','CCGG','CCTA','CCTC','CGAA','CGAC','CGAG','CGCA','CGCC','CGCG','CGGA','CGGC','CGTA','CGTC','CTAA','CTAC','CTAG','CTCA','CTCC','CTGA','CTGC','CTTA','CTTC','GAAA','GAAC','GACA','GACC','GAGA','GAGC','GATA','GATC','GCAA','GCAC','GCCA','GCCC','GCGA','GCGC','GCTA','GGAA','GGAC','GGCA','GGCC','GGGA','GGTA','GTAA','GTAC','GTCA','GTGA','GTTA','TAAA','TACA','TAGA','TATA','TCAA','TCCA','TCGA','TGAA','TGCA','TTAA']
print(len(tetramers))
rev_com =[]
for tet in tetramers:
    rev = ""
    for letter in tet:
        if letter =="A":
            rev=rev+"T"
        if letter =="C":
            rev=rev+"G"
        if letter =="T":
            rev=rev+"A"
        if letter=="G":
            rev=rev+"C"
    rev_com.append(rev)

print(rev_com)
print(tetramers)
for t in rev_com:
    if t in tetramers:
        print(t)