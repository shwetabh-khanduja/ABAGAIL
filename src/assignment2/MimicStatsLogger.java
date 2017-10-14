package assignment2;

import java.util.ArrayList;
import java.util.List;

import com.sun.tracing.dtrace.DependencyClass;

import dist.DiscreteDependencyTree;
import dist.DiscreteDependencyTreeNode;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.Instance;

public class MimicStatsLogger extends GenericStatsLogger{
	
	public MIMIC mimic;
	public List<Double> FnValues;
	public List<Double> RootNodeProbs;
	public DiscreteDependencyTree DepTree;
	public List<String> InnerNodes;
	
	public MimicStatsLogger(MIMIC mimic, int freq, DiscreteDependencyTree ddt){
		super(freq);
		this.mimic = mimic;
		FnValues = new ArrayList<Double>();
		DepTree = ddt;
		RootNodeProbs = new ArrayList<Double>();
		InnerNodes = new ArrayList<>();
	}
	
	@Override
	public boolean log(int iter, double loss, boolean force) {
		if(super.log(iter, loss, force)){
			RootNodeProbs.add(DepTree.root.probabilities[1]);
			DiscreteDependencyTreeNode node = (DiscreteDependencyTreeNode) DepTree.root.getEdge(0).getOther(DepTree.root);
			InnerNodes.add(""+node.probabilities[1][1]+","+node.probabilities[0][1]);
//			DiscreteDependencyTreeNode node2 = (DiscreteDependencyTreeNode) node.getEdge(0).getOther(node);
//			InnerNodes.add(""+node.probabilities[1][1]+","+node.probabilities[0][1]+","+node2.probabilities[1][1]);
			return true;
		}
		return false;
	}
}
