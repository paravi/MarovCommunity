import copy
import numpy as np
from numpy.random import rand
import operator
import random
import collections
from operator import itemgetter
import itertools
import time
import cPickle as pickle
import matplotlib.pyplot as plt 
import scipy
from scipy import stats
import networkx as nx
import math
import cmath
import bisect
import matplotlib
from joblib import Parallel, delayed 
import multiprocessing
import sys
from numpy import linalg as LA
import pylab
from pylab import get_current_fig_manager,show,plt,imshow
import os
import community


# The code consists of a class called RandomGraph which creates an instance of Stochastic Block Model and performs clustering based on defining a random Walk. The class has a constructor and a few functions as well which are described below.  

class RandomGraph(object):

	# constructor module
	# input: a list Chuncks which is the size of ground truth communities in the model, a parameter p which is inside cluster probability, and parameter q which is between cluster probability.
	def __init__(self,Chunks,p,q):
		self.num_nodes=sum(Chunks) # total number of nodes in the graph
		self.num_edges=0 # total number of edges in the graph
		self.node_groups=Chunks # list of the sizes for each community, say [4,5,2]
		self.node_groups_expanded=list(self.list_incr(self.node_groups)) # if chuncks is [4,5,2] we have three communities, then node_groups_expanded=[4,4+5,4+5+2]=[4,9,11]. It will be used for indexing the nodes in communities
		self.nodes=xrange(self.num_nodes) # if graph has n nodes, nodes=[0,1,..n-1] as the vertex indices
		self.p=p
		self.q=q
		self.MAX_ITERATION=1000 # max_iteration for CFTP
		self.connected=True
		self.Adj=np.zeros((self.num_nodes,self.num_nodes)) # Adjaccency matrix
		self.node_clusters=[] # If self.node_clusters=[[0,3,4],[1,2,5]] it means we have two communities on 6 nodes, first community includes [0,3,4] 
		self.TranList=[[]]
		self.TranProb=[[]]
		self.TranCumul=[[]]
		self.image_counter=0
		self.girwan_cost=0
		

		disformation_flag=0 # A flag for testing if the random graph is connected. If the instance is not connected, we repeat until we obtain a connected graph or stop
		CreationTry_thershold=20 # max number of iterations to obtain a connected graph
		c=0
		start_time = time.time()
		# print("--- %s seconds ---" % (time.time() - start_time))
		while disformation_flag==0:
			c+=1
			self.Adj=np.zeros((self.num_nodes,self.num_nodes))
			L=[0]+self.node_groups_expanded # 0 is added because indexing in Python start with 0 and work with it is much easier
			for i in xrange(0,len(L)-1):
				self.Adj[L[i]:L[i+1],L[i+1]:self.num_nodes]=copy.deepcopy(scipy.stats.bernoulli.rvs(self.q, size=(L[i+1]-L[i],self.num_nodes-L[i+1])))
				for j in xrange(L[i],L[i+1]):
					self.Adj[j,j+1:L[i+1]]=copy.deepcopy(scipy.stats.bernoulli.rvs(self.p, size=(1,-j-1+L[i+1])))

		
			self.Adj=copy.deepcopy(self.Adj+np.transpose(self.Adj))

			# print("--- %s seconds ---" % (time.time() - start_time))
			B = nx.Graph()
			B.add_nodes_from(self.nodes)
			edge_list=[]
			for r in itertools.izip(np.nonzero(self.Adj)[0],np.nonzero(self.Adj)[1]):
				#edge_list.append( ( self.nodes[r[0]], self.nodes[r[1]]) )
				B.add_edge(self.nodes[r[0]], self.nodes[r[1]])
			self.num_edges=len(B.edges())
			if nx.is_connected(B) or c>CreationTry_thershold:
				disformation_flag=1
				if c>CreationTry_thershold:
					print "Graph is disconnected"
					self.connected=False
			del B
			del edge_list
		#plt.imshow(self.Adj, cmap='flag',  interpolation='nearest')
		#plt.show()

		newAdj=copy.deepcopy(self.Adj)

		# randomly permute nodes and based on the permutation, we transform the graph and community membership 
		test=range(self.num_nodes)
		random.shuffle(test)

		LL=[0]+self.node_groups_expanded
		for i in xrange(0,len(LL)-1):
				tem=map(lambda x: test[x],range(LL[i],LL[i+1])) 
				self.node_clusters.append(tem)
		# print self.node_clusters
		

		for u in xrange(self.num_nodes):
			for v in xrange(self.num_nodes):
				newAdj[test[u]][test[v]]=self.Adj[u][v]
		self.Adj=copy.deepcopy(newAdj)
		
		#Bitmap Drawing
		#plt.imshow(self.Adj, cmap='flag',  interpolation='nearest')
	
		#plt.spy(self.Adj,cmap='Accent') #Accent
		#plt.axis('off')
		#plt.show()
		del newAdj

		#AdjArr=np.array(Adj)
		#print AdjArr

		# Count edge weights
		H = nx.Graph()
		H.add_nodes_from(self.nodes)
		for r in itertools.izip(np.nonzero(self.Adj)[0],np.nonzero(self.Adj)[1]):
			H.add_edge(self.nodes[r[0]], self.nodes[r[1]])

		partition = community.best_partition(H)
		c=[]
		for com in set(partition.values()) :
			list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == com]
			c.append(list_nodes)
		#self.draw_graph(c,None)
		self.girwan_cost=self.cluster_cost(c)

		#plt.clf()
		#nx.draw_random(H,with_labels=False, node_color='#45B97C', edge_color='black',linewidths=.001, node_size=10,alpha=1)
		#plt.show()


		#computing edge weights based on neighborhood structure
		for e in H.edges():
			H[e[0]][e[1]]['weight']= 1+pow(len(set(H.neighbors(e[0])).intersection(H.neighbors(e[1]))),3)#important factor
			#H[e[0]][e[1]]['weight']= math.exp(1*len(set(H.neighbors(e[0])).intersection(H.neighbors(e[1]))) )
			#H[e[0]][e[1]]['weight']= pow(len(set(H.neighbors(e[0])).intersection(H.neighbors(e[1]))) /float(len(H.neighbors(e[0]))*len(H.neighbors(e[1]))),6)
			#print e, H[e[0]][e[1]]['weight']


		for i in xrange(self.num_nodes-1):
			self.TranList.append([])
			self.TranProb.append([])
			self.TranCumul.append([])

		#for each node, we compute the normalized wight. For example if node 1 has three neighbors [2,3,4], then:
		# (i) 	TranList[1]=[2,3,4]
		# (ii) 	TranProb[1]=[0.1,.5,0.4]
		# (iii) TranCumul[1]=[0.1,0.6,1.0] # keeping this list will ease to perform random walk: we pick uniform random variable u say u=0.44, the find where it falls in TranCumul[1] and then perform random walk by going to   
		for v in xrange(self.num_nodes):
			q=sum([H[v][i]['weight'] for i in list(H.neighbors_iter(v))])
			for z in list(H.neighbors_iter(v)):
					self.TranList[v].append(z)
					self.TranProb[v].append(H[v][z]['weight']/float(q))
					self.TranCumul[v].append(H[v][z]['weight']/float(q))
		
		for v in xrange(self.num_nodes):
			self.TranCumul[v]=copy.deepcopy(list(self.list_incr(self.TranProb[v])))

	
	# Given [4,5,2], this function creates [4,9,11]
	def list_incr(self, iterable, func=operator.add):
		it = iter(iterable)
		total = next(it)
		yield total
		for element in it:
   			total = func(total, element)
   			yield total

	def existing_edges_inside(self,subgraph):
		z=self.Adj[subgraph]
		z=z[:,subgraph]
		return 0.5*sum(sum(z))
	def existing_edges_across(self,subgraph1,subgraph2):
		z=self.Adj[subgraph1]
		z=z[:,subgraph2]
		return sum(sum(z))


	def get_image(self, cluster):
		cluster=cluster
		test= [item for sublist in cluster for item in sublist]
		Ad=np.zeros((self.num_nodes,self.num_nodes))
		for u in xrange(len(test)):
			for v in xrange(len(test)):
				Ad[u][v]=self.Adj[test[u]][test[v]]
	
		plt.clf()
		#plt.imshow(Ad, cmap='flag',  interpolation='nearest')
		plt.spy(Ad,cmap='Accent') #Accent
		plt.axis('off')
		path='./AnimFigs'
		filename='bit%d.jpeg' % (self.image_counter)
		filename = os.path.join(path, filename)
			
		plt.savefig(filename,format='jpeg')
		plt.show()


		
	# draw a circular layout for a given clustering
	def draw_graph(self,other_cluster=None, CCPivot_cluster=None):
		path='./AnimFigs'
		alg_cluster=other_cluster
		CCPivot_cluster=CCPivot_cluster

		arraymap=np.array(self.node_clusters)
		
		community=map(lambda x:np.where(arraymap == x)[0][0], list(self.nodes))
		
		B = nx.Graph()
		edge_list=[]
		for r in itertools.izip(np.nonzero(self.Adj)[0],np.nonzero(self.Adj)[1]):
			B.add_edge(self.nodes[r[0]], self.nodes[r[1]])
		cmapp=matplotlib.colors.cnames.keys()
		#mycolors=np.random.choice(cmapp,len(self.node_clusters),replace=False)
		mycolors=['red','blue','green','orange','brown','yellow','cyan']


		OrigPos={}
		R=np.average(map(len,self.node_clusters))   
		centers=self.points(len(self.node_clusters),[0,0],.8*R)
		for w in range(len(self.node_clusters)):
			newcenter=centers[w]
			newpoints=self.points(len(self.node_clusters[w]),newcenter,1.5*R/float(len(self.node_clusters)))
			newpoints=map(tuple,newpoints)
			for v in range(len(self.node_clusters[w])):
				OrigPos.update( {self.node_clusters[w][v]:newpoints[v]} )
		nx.draw_networkx_edges(B,pos=OrigPos,width=.3,arrows=False, style='solid',alpha=1,edge_color='k')
		for w in range(len(self.node_clusters)):
			nx.draw_networkx_nodes(B,pos=OrigPos,node_size=120,nodelist=self.node_clusters[w], node_color=map(lambda x: mycolors[w],self.node_clusters[w]),font_color="k",font_size=4,width=1,alpha=1)
		#nx.draw_networkx_labels(B,pos=OrigPos, font_size=10)
		plt.figure(1)
		plt.axis('off')
		#plt.title('Original Network')

		if alg_cluster!=None:
			pos={}
			R=np.average(map(len,alg_cluster))
			centers=self.points(len(alg_cluster),[0,0],.8*R)
			for w in range(len(alg_cluster)):
				newcenter=centers[w]
				newpoints=self.points(len(alg_cluster[w]),newcenter,1.5*R/float(len(alg_cluster)))
				newpoints=map(tuple,newpoints)
				for v in range(len(alg_cluster[w])):
					pos.update( {alg_cluster[w][v]:newpoints[v]} )
			
			plt.clf()
			plt.figure(1)
			#plt.title('Our Algorithm Output')
			plt.axis('off')
			#nx.draw_networkx_edges(B,pos=pos,width=.07,arrows=False, style='solid',alpha=1,edge_color='k')
			for w in range(len(alg_cluster)):
				nx.draw_networkx_nodes(B,pos=pos,node_size=20,nodelist=alg_cluster[w], node_color=map(lambda x: mycolors[ community[x]],alg_cluster[w]),font_color="k",font_size=5,linewidths=None,alpha=1)
			#nx.draw_networkx_labels(B,pos=pos, font_size=10)
			filename='fig%d.jpeg' % (self.image_counter)
			print filename
			filename = os.path.join(path, filename)
			
			plt.savefig(filename,format='jpeg')


		# if CCPivot_cluster!=None:
		# 	CCpos={}
		# 	R=np.average(map(len,CCPivot_cluster))
		# 	centers=self.points(len(CCPivot_cluster),[0,0],.8*R)
		# 	for w in range(len(CCPivot_cluster)):
		# 		newcenter=centers[w]
		# 		newpoints=self.points(len(CCPivot_cluster[w]),newcenter,1.5*R/float(len(CCPivot_cluster)))
		# 		newpoints=map(tuple,newpoints)
		# 		for v in range(len(CCPivot_cluster[w])):
		# 			CCpos.update( {CCPivot_cluster[w][v]:newpoints[v]} )
			
		# 	plt.figure(3)
		# 	plt.title('CCPivot Output')
		# 	plt.axis('off')
		# 	#nx.draw_networkx_edges(B,pos=CCpos,width=.4,arrows=False, style='solid',alpha=1,edge_color='k')
		# 	for w in range(len(CCPivot_cluster)):
		# 		nx.draw_networkx_nodes(B,pos=CCpos,node_size=50,nodelist=CCPivot_cluster[w], node_color=map(lambda x: mycolors[ community[x]],CCPivot_cluster[w]),font_color="k",font_size=5,width=1,alpha=1)
		# 	#nx.draw_networkx_labels(B,pos=CCpos, font_size=10)



		#plt.show(block=False)
		#plt.show()


	def points(self,k,cen,r):
		p=[]
		for k in np.arange(-math.pi, math.pi, 2*math.pi/float(k)):
			p.append([cmath.rect(r,k).real,cmath.rect(r,k).imag])
		return map(lambda x: [x[0]+cen[0],x[1]+cen[1]],p)
	def list_incr(self,iterable, func=operator.add):
	    it = iter(iterable)
	    total = next(it)
	    yield total
	    for element in it:
	        total = func(total, element)
	        yield total

	# given a node by state id "state" and  uniform random variable "ran", we find a random neighbor
	def OneStepTransit(self,state,ran):
		i=bisect.bisect_left(self.TranCumul[state],ran)
		return self.TranList[state][i]

	# computing edit cost (the number of errroneous edges) for a given partitioning of nodes
	def cluster_cost(self,cluster):
		induced_graph=np.zeros((self.num_nodes,self.num_nodes))
		for w in cluster:
			if len(w)>1:
				for e in itertools.combinations(w, 2):
					induced_graph[e[0],e[1]]=1
					induced_graph[e[1],e[0]]=1
		return 0.5*(np.sum(np.not_equal(self.Adj,induced_graph)))


	# perform CFTP and record clusters along the critical time (refer to paper for critical times)
	def BackwardPath(self):
		u=[]
		flag=0
		counter=0
		coelsce_tracker_users=[]
		critical_times=[]
		critical_times_cost=[]
		optimal_cluster=[]
		optimal_cost=1e300

		#start_time = time.time()
		
		while flag==0:
			current_state=copy.deepcopy(self.nodes)
			interval=counter+1# 2**counter
			w=np.random.uniform(0,1,1)
			u.insert(0,w[0])
			counter=counter+1
			
			#print("--- %s K seconds ---" % (time.time() - start_time))
			for k in range(0, interval):
				current_state=copy.deepcopy(map (lambda x: self.OneStepTransit (x,u[k]),current_state))

			if interval==1:	
				A=list(enumerate(current_state))
				AA=set(current_state)
				critical_times.append(interval)
				temp0=[]
				for v in AA:
					temp0.append([indx for (indx,cr) in A if cr==v])
				coelsce_tracker_users.insert(0,temp0)
				current_cost=self.cluster_cost(coelsce_tracker_users[0])
				critical_times_cost.append(current_cost)
				if current_cost<optimal_cost:
					optimal_cost=current_cost
					optimal_cluster=copy.deepcopy(coelsce_tracker_users[0])
				


			else:
				recent_critical_time=critical_times[-1]
				recent_cluster=copy.deepcopy(coelsce_tracker_users[-1])
				recent_cluster_index=range(len(recent_cluster))
				temp=[]

				for i in recent_cluster_index:
					if len(set([current_state[j] for j in recent_cluster[i]]))==1:
						temp.append((i, list(set([current_state[j] for j in recent_cluster[i]]))[0]))
				# print temp
				
				Z=collections.Counter(map(lambda x: itemgetter(1)(x),temp))
				

				final_values=[a for (a,b) in Z.items() if b>=2]	
				# print Z
				# print final_values
				# print "==========" 

				equal_count_indicator=[]*len(final_values)
				coelsce_tracker_users.append([])
				for r in final_values:
					clusters_coelesced=filter(lambda x : x[1]==r,temp)
					clusters_coelesced_index=map( lambda x: itemgetter(0)(x),clusters_coelesced)
					# print clusters_coelesced
					# print clusters_coelesced_index
					
			
					G_index= list(itertools.chain(*[recent_cluster[x] for x in clusters_coelesced_index]))
					G_state=copy.deepcopy(G_index)
					# print G_state
					
					current_G_state=copy.deepcopy(G_state)


					mycount=[1]*len(current_G_state)
					for k in range(0, interval):
						current_G_state=copy.deepcopy(map (lambda x: self.OneStepTransit (x,u[k]),current_G_state))
						for j in range(len(current_G_state)):
							if current_G_state[j] in G_state: 
								mycount[j]+=1
								
		
					
					equal_count_indicator.append(mycount.count(mycount[0]) == len(mycount))
		
					temp2=[]
					if equal_count_indicator[-1]==True:
						for s in clusters_coelesced_index:
							temp2.append(recent_cluster[s])
	
							
							recent_cluster_index.remove(s)

	
						temp2 = list(itertools.chain(*temp2))
						coelsce_tracker_users[-1].append(temp2)
					
				for r in recent_cluster_index:
					coelsce_tracker_users[-1].append(recent_cluster[r])
				del coelsce_tracker_users[0]
				

				if any(equal_count_indicator):
					critical_times.append(interval)
					# print coelsce_tracker_users[0]
					# print
					# print len(coelsce_tracker_users[0])
					# print
					self.image_counter+=1
					self.draw_graph(coelsce_tracker_users[0],None)
					#self.get_image(coelsce_tracker_users[0])
					
					
					#print("--- %s G seconds ---" % (time.time() - start_time))
					current_cost=self.cluster_cost(coelsce_tracker_users[0])
					#print("--- %s F seconds ---" % (time.time() - start_time))
					critical_times_cost.append(current_cost)
					if current_cost<optimal_cost:
						optimal_cost=current_cost
						optimal_cluster=copy.deepcopy(coelsce_tracker_users[0])
						

					#print coelsce_tracker_users
			


			user_coel_index= (len(set(current_state)) ==1)
			
			if (interval-critical_times[-1]>=100 or (user_coel_index)):
				
				flag=1
				
				# if user_coel_index:
				# 	print "Coelsce happend in " +str(interval) +" steps."
				# 	print critical_times
				# if interval>=self.MAX_ITERATION:
				# 	print "Maximum iteration reached."
				# if interval-critical_times[-1]>=100:
				# 	print "Difference between critical times exceeds 100"
				# 	print critical_times
					# print critical_times_cost
		#print optimal_cluster
		return optimal_cost,optimal_cluster

	# Perform CCPivot Algorithm on the graph
	def CCPivot(self):
		#user_indexes=range(self.num_nodes)
		cluster=[]
		container_users=range(self.num_nodes)
		
		while len(container_users)!=0:
			k=random.choice(container_users)
			container_users.remove(k)
			cluster.append([k])
			
			for j in sorted(container_users,reverse=True):
				
				if j in self.TranList[k]:
					cluster[-1].append(j)
					container_users.remove(j)
		#print("--- %s seconds ---" % (time.time() - start_time))
		#print "g"
		# print "====================================================================="
		#print "CCPivot output cluster is: "+str(cluster)
		induced_graph=np.zeros((self.num_nodes,self.num_nodes))
		for w in cluster:
			if len(w)>1:
				for e in itertools.combinations(w, 2):
					induced_graph[e[0],e[1]]=1
					induced_graph[e[1],e[0]]=1
		return 0.5*(np.sum(np.not_equal(self.Adj,induced_graph))),cluster

	
	def ComapreCost(self):
		return self.BackwardPath()[0] , self.CCPivot()[0], self.girwan_cost

start_time = time.time()


G=RandomGraph([40]*5,.65,.05)
# print "girwan_cost: "+str(G.girwan_cost)
# print "cc pivot cost: "+str(G.CCPivot()[0])
# print "our cost: "+ str(G.BackwardPath()[0])






G.draw_graph()
#G.get_image(G.MCL(0.01,2,2))
#
#G.get_image(G.CCPivot()[1])
#G.BackwardPath()
#print G.BackwardPath()[0]
#print G.CCPivot()[0]
G.BackwardPath()
#G.draw_graph(G.BackwardPath()[1] ,G.CCPivot()[1])
#G.draw_graph(G.BackwardPath()[1] ,G.CCPivot()[1])
# time.sleep(1000)

# G=RandomGraph([300,300,300,500,120,320,550,440,780],.7,0.2)
# print("--- %s seconds ---" % (time.time() - start_time))

# print G.CCPivot()[0]
# print("--- %s seconds ---" % (time.time() - start_time))

# print G.BackwardPath()[0]
# print("--- %s seconds ---" % (time.time() - start_time))



