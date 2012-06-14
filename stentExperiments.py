""" Module stentExperiments

To evaluate stent segmentation methods using the pirt.experiment framework.

"""

# todo: if experiment.py were in pyzolib, maybe we would not need pirt.
from pirt import experiment, ssdf

import sys, os, time
import numpy as np
import visvis as vv
from pypoints import Point, Pointset, Aarray
import from visvis.utils import graph
import . stentGraph
import stentDirect


## Init db, cylinder sets and some globals
# patnr's per stent:
# - aneurx: [10,11,12,14]
# - talent: [21, 22]
# - zenith: [1,2,3,4,5,6,7,8,18]
# - excluder: [9,13,15,16]
# - annaconda: [20,28]

# Define home directory
homeDir = '/home/almar/'
if sys.platform.startswith('win'):
    homeDir = 'c:/almar/'

# Get database with annotations and sort
annDb = ssdf.load(homeDir + 'data/misc/unfoldedDB.ssdf')
def sorter(x):            
    return '%s%02i%03i' % (x.stentType, x.patnr, x.im.shape[0])
annDb.thelist.sort(key=sorter)

# Create sets per stent type
ANEURXSET = set()
ZENITHSET = set()
TALENTSET = set()
for i in range(len(annDb.thelist)):
    st = annDb.thelist[i].stentType.lower()
    if st == 'aneurx':  ANEURXSET.add(i)
    if st == 'zenith':  ZENITHSET.add(i)
    if st == 'talent':  TALENTSET.add(i)

# Define test and training indices
TRAININGCYLINDERS = [0,3,6,7,10,13,16,19] # 21 sucks
TESTCYLINDERS = [1,2,4,5,8,9,11,12,14,15,17,18,20,22]

# Select stent types to consider
tmp = set.union(ANEURXSET, ZENITHSET)
TRAININGCYLINDERS = sorted(list( set(TRAININGCYLINDERS).intersection(tmp) ))
TESTCYLINDERS = sorted(list( set(TESTCYLINDERS).intersection(tmp) ))

# Patient numbers used in training and test set
PATNRS_TRAININGCYLINDERS = set([annDb.thelist[i].patnr for i in TRAININGCYLINDERS])
PATNRS_TESTCYLINDERS = set([annDb.thelist[i].patnr for i in TESTCYLINDERS])

# For visualization
STENTTYPELEGEND = {'aneurx':('r',':'), 'zenith':('b','--'), 'talent':('y', '.-')}


class StentExperiment(experiment.Experiment):
    """ StentExperiment(params, database=None)
    
    Experiment class for stent segmentation.
    To use this class, override the _getGraphByAlg() method.
    
    For tuning, the experiment series should be as follows:
      * Series 0: Alg and annotators
      * Series 1: Parameter to change (makes it possible to buffer alg results)
      * Series 2: Cylinders
    
    For testing, the experiment series should be as follows:
      * Series 0: Alg and annotators
      * Series 1: Cylinders
    
    """
    
    # "overload" this class attribute
    XLABELS = { 'nphases': '\i{n_{phases}} (Number of phases to average)',
        }
    
    
    
    def _getCylinderFromNumber(self, nr):
        """ _getCylinderFromNumber(nr)
        
        Get the cylinder info corresponding with the given cylinder number.
        
        """
        return annDb.thelist[nr]
    
    
    def _loadVolume(self, cylinder, nphases):
        """ _loadVolume(cylinder, nphases)
        
        Load a volume, given a cylinder. Also specify amount of phases.
    
        """
        return stentDirect.loadVol(cylinder.patnr, nphases)


    def _cropToCylinder(self, graph1, cylinder):    
        """ _cropToCylinder(graph, cylinder)
        
        Given a graph and the description of a cylinder, will return a new
        graph with only those nodes that lie inside the given cylinder.
        
        """
        
        # Init cropped graph
        newGraph = graph.Graph()
        
        # First make a copy, but mark nodes that are outside the cylinder    
        for node in graph1:
            tmp = self._isPointInCylinder(cylinder, node)
            if tmp:
                newGraph.AppendNode(node.copy())
                newGraph[-1].dontCare = {1:True, 2:False}[tmp]
            else:
                newGraph.AppendNode(Point(0,0,0))
        
        # Establish the connections    
        for c in graph1.GetEdges():
            newGraph.CreateEdge(c._i1, c._i2)
        
        # Remove marked nodes
        for node in [node for node in newGraph]:
            if node.norm() == 0:
                newGraph.Remove(node)
        
        # Done!
        return newGraph
    
    
    def _isPointInCylinder(self, cylinder, p, margin=0.5, radiusMargin=1.0):        
        """ _isPointInCylinder(cylinder, p)
        
        Function to determine whether the given point lies inside the given 
        cylinder. Used to crop the graph to a cylinder.
        
        Returns 0 if point is outside the cylinder, returns 1 if inside but
        within margin, and 2 if fully inside.
        There is 1 mm radius margin.
        
        """
        
        # Calc points and vectors
        a = Point(cylinder.pos1)
        b = Point(cylinder.pos2)
        ab = a-b
        ap = a-p
        
        # It can be shown that ...
        t = ab.dot(ap) / ab.norm()**2
        t = float(t)
        
        # So we can calculate q
        q = a + t*(b-a)
        
        # Calculate margin
        maxt = ab.norm()
        t *= maxt
        
        # Test whether p was inside the specified cylinder
        radius = cylinder.radius + radiusMargin
        if q.distance(p) > radius:
            return 0
        elif t < -margin or t > maxt+margin:
            return 0
        elif t < margin or t > maxt-margin:
            return 1
        else:
            return 2        
        
    
    def _getGraphByAnnotator(self, cylinderNr, annotator):
        """ _getGraphByAnnotator(cylinderNr, annotator)
        
        Get the 3D graph of the given cylinder as annotated by        
        the specified annotator.
        
        """
        
        # Get cylinder
        cylinder = annDb.thelist[cylinderNr]
        
        # Obtain 2D graph
        graph2 = graph.Graph()
        graph2.Unpack( cylinder.annotations[annotator] )
        
        # Get maps with coordinates
        imx, imy, imz = cylinder.imx, cylinder.imy, cylinder.imz
        
        # Create 3D graph from 2D graph
        graph3 = graph.Graph()
        for node in graph2:
            i, j = int(node.y+0.5), int(node.x+0.5)
            try:
                p = Point(imx[i,j], imy[i,j], imz[i,j])
            except IndexError:
                p = Point(0,0,0)
            graph3.AppendNode(p)        
        
        # Make connections
        for c in graph2.GetEdges():
            graph3.CreateEdge(c._i1, c._i2)
        
        # Done!
        return graph3
    
    
    
    def _getGraphByAlg(self, cylinderNr, params):
        """ _getGraphByAlg(cylinderNr, params)
        
        Get the 3D graph as produced by the algorithm using the specified
        parameters. If necessary, the algorithm is run to provide that graph.
        An sd object can be specified to run experiments quicker by avoiding
        to re-apply the whole algorithm.
        
        Returns the graph.
        
        Overload this method.
        
        """
        raise NotImplementedError()
    
    
    
    def experiment(self, params):
        """ experiment(params)
        
        Experiment method.
        
        We employ the following experiment parameters:            
        exp_annotator : None or string
            None means the algorithm. When a string, it signifies the 
            annotator name.
        exp_cylinder : int
            The cylinder number to test.
        
        """
                        
        # Get cylinder nr
        cylinderNr = params.exp_cylinder
        
        # Get graph1
        if params.exp_annotator is None:
            graph = self._getGraphByAlg(cylinderNr, params)
        else:
            # Get annotated graph
            graph = self._getGraphByAnnotator(cylinderNr, params.exp_annotator)
            self._save_next_result = False
        
        # Done
        return graph
    
    
    def quantify_results_tuning(self, annotatorId1, annotatorId2, margin=3.0):
        """ quantify_results(annotatorId1, annotatorId2, margin=3.0)
        
        This method does the comparing of the graphs. A tuning experiment
        should have been performed:
          * series 0: alg and annotators
          * series 1: parameter value being tuned
          * series 2: cylinders
        
        annotatorId1 and annotatorId2 are the indices of series 0. Per 
        convention, 0 means algorithm, higher values signify the annotators.
        
        Returns a dict with lists of matching scores for each stent type (and one
        for total).
        
        """
        
        # Get parameter ranges
        param, values_tune = self.get_series_params(1)
        param, values_cylinder = self.get_series_params(2)
        
        # Init matching scores dict. For each stent type and for the total
        N = len(values_tune)
        matchingScores = {} 
        matchingScores['total'] = []
        
        
        for j in range(len(values_cylinder)):
            
            # Make sure the matching score dict has entry for this stent type
            cylinder = annDb.thelist[values_cylinder[j]]
            stentType = cylinder.stentType.lower()
            if not stentType in matchingScores:
                matchingScores[stentType] = []
            
            # Get lists of matching scores
            ms_list_total = matchingScores['total']
            ms_list_stentType = matchingScores[stentType]
            
            for i in range(N):
                
                # Get graphs 
                graph1 = self.get_result(annotatorId1, i, j)
                graph2 = self.get_result(annotatorId2, i, j)
                
                # Compare them to obtain matching score
                ms = graph.compareGraphs(graph1, graph2, margin)
                
                # Make sure the lists are long enough (append null-matchingscore)
                for ms_list in [ms_list_total, ms_list_stentType]:
                    if len(ms_list) <= i:
                        ms_list.append(graph.MatchingScore(0,0,0))
                
                # Contribute this score to the scores
                ms_list_total[i] += ms
                ms_list_stentType[i] += ms
        
        # Done
        return matchingScores
        
    
    def quantify_results_performance(self, annotatorId1, annotatorId2, margin=3.0):
        """ quantify_results(annotatorId1, annotatorId2, margin=3.0)
        
        This method does the comparing of the graphs. A performance
        experiment should have been performed:
          * series 0: alg and annotators
          * series 1: cylinders 
        
        AnnotatorId1 and annotatorId2 are the indices of series 0. Per 
        convention, 0 means algorithm, higher values signify the annotators.
        
        Returns a dict with a matching score for each stent type (and one
        for total).
        
        """
        
        # Get parameter range
        param, values_cylinder = self.get_series_params(1)
        
        # Init matching scores dict. For each stent type and for the total
        matchingScores = {} 
        matchingScores['total'] = graph.MatchingScore(0,0,0)
        
        
        for j in range(len(values_cylinder)):
            
            # Make sure the matching score dict has entry for this stent type
            cylinder = annDb.thelist[values_cylinder[j]]
            stentType = cylinder.stentType.lower()
            if not stentType in matchingScores:
                matchingScores[stentType] = graph.MatchingScore(0,0,0)
            
            # Get graphs 
            graph1 = self.get_result(annotatorId1, j)
            graph2 = self.get_result(annotatorId2, j)
            
            # Compare them to obtain matching score
            ms = graph.compareGraphs(graph1, graph2, margin)
            
            # Contribute this score to the scores
            matchingScores['total'] += ms
            matchingScores[stentType] += ms
        
        # Done
        return matchingScores
        
    
    def show_results_tuning(self, margin=3.0):
        """ show_results_tuning(margin=3.0)
        
        Show the results of the tuning experiments.
        
        This method does the comparing of the graphs. A tuning experiment
        should have been performed:
          * series 0: alg and annotators
          * series 1: parameter value being tuned
          * series 2: cylinders
        
        """
        
        # Get params
        param_annotators, values_annotators = self.get_series_params(0)
        param_param, values = self.get_series_params(1)
        
        # Get results of alg compared to each annotator
        resultsPerAnnotator = [] # list of dicts with lists
        for i in range(1, len(values_annotators)):            
            matchingScores = self.quantify_results_tuning(0, i, margin)
            resultsPerAnnotator.append(matchingScores)
        
        # Create combined total of all annotators
        N = len( matchingScores['total'] )
        totalTotal = []
        for i in range(N):
            ms = graph.MatchingScore(0,0,0) 
            for tmp in resultsPerAnnotator:
                ms += tmp['total'][i]
            totalTotal.append(ms)
        
        # Init figure and axes
        fig = vv.figure()
        a1 = vv.subplot(1,1,1)
        
        # Draw results
        if True:            
            
            # Show average of total
            scores = [mc.val*100 for mc in totalTotal]
            l = vv.plot(values, scores, 
                mc='k', lc='k', lw=3, ms='.', mw=7, mew=0)
            l._points[:,2] += 0.1 # make total on top
            
            # Per annotator ...
            for i in range(len(resultsPerAnnotator)):
                matchingScores = resultsPerAnnotator[i]
                
                # Show average of each stent type ...
                for stentType in sorted( matchingScores.keys() ):
                    if stentType == 'total':
                        continue
                    scores = [mc.val*100 for mc in  matchingScores[stentType]]
                    if scores[0] == -100:
                        continue
                    c, ls = STENTTYPELEGEND[stentType]
                    vv.plot(values, scores, 
                        mc=c, lc=c, ls=ls, lw=2, ms='.', mw=0, mew=0)
            
        # Set labels etc
        if param_param in self.XLABELS:
            vv.xlabel(self.XLABELS[param_param])
        else:            
            vv.xlabel(param_param)
        vv.ylabel('DSC [%]')
        a1.axis.showGrid = 1
        a1.SetLimits(rangeY=(0,100))
        a1.legend = 'Total', 'Aneurx', 'Zenith'
        
        # Scale fonts
        fig.relativeFontSize = 1.3
        a1.position.Correct(dh=-5)
    
    
    def show_results_annotators(self, margin=3.0):
        """ show_results_performance(self, margin=3.0)
        
        Show the performance of the algorithm.
        
        This method does the comparing of the graphs. A performance
        experiment should have been performed:
          * series 0: alg and annotators
          * series 1: cylinders 
        
        """
        
        # Get params
        param, values_annotators = self.get_series_params(0)
        
        # Get results of alg compared to each annotator
        resultsPerAnnotator = [] # list of dicts with lists
        for i in range(1, len(values_annotators)):            
            matchingScores = self.quantify_results_performance(0, i, margin)
            resultsPerAnnotator.append(matchingScores)
        
        # Get matching score per annotator
        resultsPerAnnotatorPair = []
        annotatorPairTicks = []
        for i in range(1, len(values_annotators)):
            for j in range(i+1,len(values_annotators)):
                ann1, ann2 = values_annotators[i], values_annotators[j]
                matchingScores = self.quantify_results_performance(i,j, margin)
                resultsPerAnnotatorPair.append(matchingScores)
                annotatorPairTicks.append('obs%i vs obs%i' % (i,j))
        
        
        # Init bar properties
        colors = []
        values = []
        xx = []
        
        # Set bar properties
        barDist = 0.4 
        barStride = 0.2
        barWidth = barDist * 0.85
        x = -barDist/2.0
                
        # Calculate inter-observer values
        for annCount in range(len(resultsPerAnnotatorPair)):
            keys = resultsPerAnnotatorPair[annCount].keys()
            keys.sort(); 
            for stentType in keys:
                if stentType == 'total':
                    continue
                ms = resultsPerAnnotatorPair[annCount][stentType]
                values.append( 100 * ms.val )        
                colors.append( STENTTYPELEGEND[stentType.lower()][0] )
                xx.append(x)
                x += barDist
            x += barStride
        
        # Draw
        xTicks = annotatorPairTicks
        self._show_results_for_performance_and_annotators(
            xx, values, colors, xTicks, barWidth, keys)
    
    
    def show_results_performance(self, margin=3.0):
        """ show_results_performance(self, margin=3.0)
        
        Show the performance of the algorithm.
        
        This method does the comparing of the graphs. A performance
        experiment should have been performed:
          * series 0: alg and annotators
          * series 1: cylinders 
        
        """
        
        # Get params
        param, values_annotators = self.get_series_params(0)
        
        # Get results of alg compared to each annotator
        resultsPerAnnotator = [] # list of dicts with lists
        for i in range(1, len(values_annotators)):            
            matchingScores = self.quantify_results_performance(0, i, margin)
            resultsPerAnnotator.append(matchingScores)
        
        # Init bar properties
        colors = []
        values = []
        xx = []
        
        # Set bar properties
        barDist = 0.4 
        barStride = 0.2
        barWidth = barDist * 0.85
        x = -barDist/2.0
        
        # Calculate alg performance values
        for annCount in range(len(resultsPerAnnotator)):
            keys = resultsPerAnnotator[annCount].keys()
            keys.sort()
            for stentType in keys:
                if stentType == 'total':
                    continue
                ms = resultsPerAnnotator[annCount][stentType]
                values.append( 100*ms.val )
                colors.append( STENTTYPELEGEND[stentType.lower()][0] )
                xx.append(x)
                x += barDist
            x += barStride
        
        # Draw
        xTicks = ['alg vs obs%i'%(i+1) for i in range(len(values_annotators))]
        self._show_results_for_performance_and_annotators(
            xx, values, colors, xTicks, barWidth, keys)
    
    
    def show_results_performance_table(self, margin=3.0):
        """ show_results_performance_table(self, margin=3.0)
        
        Show the performance of the algorithm by printing true positives,
        false positives and false negatives to the screen.
        
        This method does the comparing of the graphs. A performance
        experiment should have been performed:
          * series 0: alg and annotators
          * series 1: cylinders 
        
        """
        
        # Get params
        param, values_annotators = self.get_series_params(0)
        
        # Get results of alg compared to each annotator
        resultsPerAnnotator = [] # list of dicts with lists
        for i in range(1, len(values_annotators)):            
            matchingScores = self.quantify_results_performance(0, i, margin)
            resultsPerAnnotator.append(matchingScores)
        
        # Calculate alg performance values
        for annCount in range(len(resultsPerAnnotator)):
            keys = resultsPerAnnotator[annCount].keys()
            keys.sort()
            print 'annotator', annCount
            for stentType in keys:
                if stentType == 'total':
                    continue
                ms = resultsPerAnnotator[annCount][stentType]
                val = ms.val*100
                print stentType, ms.TP, 'TP ', ms.FP, 'FP ', ms.FN, 'FN ', val, 'score'
    
    
    def _show_results_for_performance_and_annotators(self, xx, values, 
            colors, xTicks, barWidth, keys):
        """ Helper function.        
        """
        
        # Init figure and axes
        f = vv.figure();
        a1 = vv.subplot(111)
        
        # Create bar plot
        b = vv.bar3(xx, values, width=barWidth)
        b.colors = colors
        a1.axis.xTicks = xTicks
        
        # Apply text to each bar
        for i in range(len(values)):
            val = float(values[i])            
            t = vv.Text(b, '\b{%2.1f}'%val, xx[i], -barWidth*0.7, val-4)
            t.halign = 0
            t.textColor = 'w'
        
        # Set labels etc        
        vv.zlabel('DSC [%]')
        a1.axis.showGridZ = 1
        a1.axis.yTicks = []                
        a1.SetLimits(rangeZ=(0,100), margin=0.02)
        
        # Set view
        a1.daspect = 1,1,0.017
        a1.axis._minTickDist = 20
        a1.camera.view_zoomx = a1.camera.view_zoomy = 2.55
        a1.camera.view_az = -20
        a1.camera.view_el = 20
        fig = vv.gcf()
        fig.relativeFontSize = 1.3
        
        # Enable a legend
        for stentType in keys:
            if stentType == 'total':
                continue
            clr = STENTTYPELEGEND[stentType.lower()][0]
            l=vv.plot([0],[0], ls='', ms='s', mc=clr, axes=a1, axesAdjust=0)
            l.visible = False
        a1.legend = 'Aneurx', 'Zenith'

