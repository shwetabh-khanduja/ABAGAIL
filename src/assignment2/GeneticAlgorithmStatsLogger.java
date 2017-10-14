package assignment2;

import java.util.ArrayList;
import java.util.List;

import opt.ga.GeneticAlgorithmProblem;
import opt.ga.StandardGeneticAlgorithm;
import shared.Instance;

public class GeneticAlgorithmStatsLogger extends GenericStatsLogger{
	
	public StandardGeneticAlgorithm StandardGa;
	public List<Double> FnValues;
	
	public GeneticAlgorithmStatsLogger(StandardGeneticAlgorithm ga, int freq){
		super(freq);
		StandardGa = ga;
		FnValues = new ArrayList<Double>();
	}
	
	@Override
	public boolean log(int iter, double loss, boolean force) {
		if(super.log(iter, loss, force)){
				if(StandardGa != null){
				Instance i = StandardGa.getOptimal();
				Double value = ((GeneticAlgorithmProblem)StandardGa.getOptimizationProblem()).value(i);
				FnValues.add(value);
			}
			return true;
		}
		return false;
	}
}
