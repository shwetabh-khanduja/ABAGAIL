package assignment2;

import java.io.IOException;
import java.io.File;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.Random;
import java.util.function.Function;

import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;
import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.SwapNeighbor;
import opt.example.CountOnesEvaluationFunction;
import opt.example.FlipFlopEvaluationFunction;
import opt.example.FourPeaksEvaluationFunction;
import opt.example.KnapsackEvaluationFunction;
import opt.example.TravelingSalesmanCrossOver;
import opt.example.TravelingSalesmanRouteEvaluationFunction;
import opt.example.TravelingSalesmanSortEvaluationFunction;
import opt.example.TwoColorsEvaluationFunction;
import opt.ga.CrossoverFunction;
import opt.ga.SingleCrossOver;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.SwapMutation;
import opt.ga.UniformCrossOver;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import shared.Trainer;
import shared.writer.CSVWriter;

public class Runs {
	public static void main(String[] args) throws Exception{
//		args : C:/Users/shkhandu/OneDrive/Gatech/Courses/ML/Assignment2 ropt nnet output_test_lr_10k-iters_30-hiddenlayers_all_1.csv 20,30,40,50,60,70,80,90,100
		String rootPath = args[0];
		System.out.println("rootpath: " + args[0]);
		System.out.println(args[1] + " " + args[1].length());
		if(args[1].equals("nnet")){
			NeuralNetworkExperiments.Perform(rootPath,args[2],args[3]);
		}
		else if(args[1].equals("ropt")){
			int runs = Integer.parseInt(args[2]);
			System.out.println("Randomised optimizations with total runs : " + runs);
			RunOnOptimizationProblems(rootPath, runs);	
		}
	}
	
	public static void RunOnOptimizationProblems(String rootPath, int runs) throws IOException{

		int[] seeds = new int[runs];
		for(int i=0; i< runs; i++){
			seeds[i] = i;
		}
		
		boolean[] computeForAlgos = new boolean[]{true,true,true,true};
		for (int seed : seeds) {
			Function<Object[],StatsLogger[]> f;
			
			System.out.println("knapsack");
			String knapsackOutputPath = rootPath + "/knapsack/" + seed;
			CreateDirectoryIfNotExists(knapsackOutputPath);
			f = a -> GetResultsForKnapSackProblemWithSize((int)a[0], (int)a[1], (HashMap<String,Long[]>)a[2], (boolean[])a[3]);
			GetResults(knapsackOutputPath, seed, new int[]{30,50,70,100,130}, f, computeForAlgos);

			System.out.println("TwoColors");
			String twoColorsOutputPath = rootPath + "/twocolors/" + seed;
			CreateDirectoryIfNotExists(twoColorsOutputPath);
			f = a -> GetResultsForTwoColorsProblemWithSize((int)a[0], (int)a[1], (HashMap<String,Long[]>)a[2], (boolean[])a[3]);
			GetResults(twoColorsOutputPath, seed, new int[]{20,50,80,100,120}, f, computeForAlgos);
			
			System.out.println("countones");
			String countOnesPath = rootPath + "/countones/" + seed;
			CreateDirectoryIfNotExists(countOnesPath);
			f = a -> GetResultsForCountOnesProblemWithSize((int)a[0], (int)a[1], (HashMap<String,Long[]>)a[2], (boolean[])a[3]);
			GetResults(countOnesPath, seed, new int[]{30,40,50,60,70}, f, computeForAlgos);
		}
	}
	
	public static void GetResults(
			String outputPath, 
			int seed,
			int[] sizes,
			Function<Object[], StatsLogger[]> f,
			boolean[] computeForAlgos) throws IOException{
		
		String rhcOutput = outputPath + "/rhc.csv";
		String saOutput = outputPath + "/sa.csv";
		String gaOutput = outputPath + "/ga.csv";
		String mimicOutput = outputPath + "/mimic.csv";
		String timesOutput = outputPath + "/times.csv";
		String fnevalOutput = outputPath + "/fnevals.csv";
		String[] files = new String[]{rhcOutput,saOutput,gaOutput,mimicOutput};
		
		CSVWriter[] writers = new CSVWriter[]{
				!computeForAlgos[0] ? null : new CSVWriter(rhcOutput, "size,iters,fn_value".split(",")),
				!computeForAlgos[1] ? null : new CSVWriter(saOutput, "size,iters,temp,fn_value".split(",")),
				!computeForAlgos[2] ? null : new CSVWriter(gaOutput, "size,iters,fn_value".split(",")),
				!computeForAlgos[3] ? null : new CSVWriter(mimicOutput, "size,iters,fn_value,root_node_prob,n1,n2,n3".split(","))
		};
		
		CSVWriter timesWriter = new CSVWriter(timesOutput, "id,size,rhc,sa,ga,mimic".split(","));
		CSVWriter fnevalWriter = new CSVWriter(fnevalOutput, "id,size,rhc,sa,ga,mimic".split(","));
		timesWriter.open();
		fnevalWriter.open();
		
		for(int i = 0; i < writers.length;i++){
			if(writers[i] == null) continue;
			new PrintWriter(files[i],"UTF-8").close();
			writers[i].open();
		}
		HashMap<String, Long[]> times = new HashMap<>();
		for(int i = 0; i < sizes.length; i++)
		{
			times.clear();
			int size = sizes[i];
			StatsLogger[] loggers = f.apply(new Object[]{size, seed, times, computeForAlgos});
			Long[] rhcTime = times.getOrDefault("rhc", new Long[]{(long) -1,(long) -1});
			Long[] saTime = times.getOrDefault("sa", new Long[]{(long) -1,(long) -1});
			Long[] gaTime = times.getOrDefault("ga", new Long[]{(long) -1,(long) -1});
			Long[] mimicTime = times.getOrDefault("mimic", new Long[]{(long) -1,(long) -1});
			String timeline = seed + "," + size + "," + rhcTime[0] + "," + saTime[0] +"," + gaTime[0] + "," + mimicTime[0];
			timesWriter.write(timeline);
			timesWriter.nextRecord();
			String evalline = seed + "," + size + "," + rhcTime[1] + "," + saTime[1] +"," + gaTime[1] + "," + mimicTime[1];
			fnevalWriter.write(evalline);
			fnevalWriter.nextRecord();
			
			for(int j = 0; j < writers.length;j++)
			{
				if(loggers[j] == null){
					continue;
				}
				
				if(j == 1)
				{
					SimulatedAnnealingStatsLogger logger = (SimulatedAnnealingStatsLogger)loggers[j];
					for(int k = 0; k < logger.Count(); k++)
					{
						String line = size + ","+ (logger.Iters.get(k))+","+logger.Temperatures.get(k)+","+logger.Loss.get(k);
						writers[j].write(line);
						writers[j].nextRecord();
						
					}
				}
				else if(j == 2)
				{
					GeneticAlgorithmStatsLogger logger = (GeneticAlgorithmStatsLogger) loggers[j];
					for(int k = 0; k < logger.Count(); k++)
					{
						String line = size + ","+ (logger.Iters.get(k)) +","+logger.FnValues.get(k);
						writers[j].write(line);
						writers[j].nextRecord();
					}
				}
				else if(loggers[j] instanceof MimicStatsLogger){
					MimicStatsLogger msLogger = (MimicStatsLogger)loggers[j];
					for(int k = 0; k < msLogger.Count(); k++)
					{
						String line = size + ","+ (msLogger.Iters.get(k)) +","+msLogger.Loss.get(k)+","+msLogger.RootNodeProbs.get(k)+","+msLogger.InnerNodes.get(k);
						writers[j].write(line);
						writers[j].nextRecord();
					}
				}
				else
				{
					GenericStatsLogger logger = (GenericStatsLogger) loggers[j];
					for(int k = 0; k < logger.Count(); k++)
					{
						String line = size + ","+ (logger.Iters.get(k)) +","+logger.Loss.get(k);
						writers[j].write(line);
						writers[j].nextRecord();
					}
				}
			}
		}
		for (CSVWriter writer : writers) {
			if(writer == null) continue;
			writer.close();
		}
		timesWriter.close();
		fnevalWriter.close();
	}
	
	public static StatsLogger[] GetResultsForFourPeaksProblemWithSize(int size, int seed, HashMap<String, Long[]> times, boolean[] AlgosToRun){
		int N = size;
		int T = (int) (0.10 * N);
		int maxIters = 1200000;
		int iterTol = 1200000; // iters to wait before concluding convergence
		double convergenceErrTol = 1; // fn value should always increase by atleast this amount in each iter
		int loggingFreq = 50;
		StatsLogger[] results = new StatsLogger[4];
		Distribution.random.setSeed(seed);
		
		EvaluationFunction _ef = new FourPeaksEvaluationFunction(T);
		EvaluationFunctionWithLogging ef = new EvaluationFunctionWithLogging(_ef);
		
		int[] ranges = GetArray(N, 2);
		DiscreteUniformDistribution uniformDist = new DiscreteUniformDistribution(ranges);
		NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
		HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, uniformDist, nf);
		GenericStatsLogger genericLogger = new GenericStatsLogger(loggingFreq);
		
		RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
		ConvergenceTrainerWithStatsLogging ctRhc = new ConvergenceTrainerWithStatsLogging(rhc, convergenceErrTol, maxIters, genericLogger, iterTol);
		if(AlgosToRun[0]){
			ef.Reset();
			Long[] res = MeasureTime(ctRhc);
			res[1] = ef.Count;
			times.put("rhc", res);
			System.out.println("Done rhc");
			results[0] = ctRhc.Logger;
		}
		
		double initialTemp = 1e11;
		double coolingRate = 0.99;
		SimulatedAnnealing sa = new SimulatedAnnealing(initialTemp, coolingRate, hcp);
		SimulatedAnnealingStatsLogger logger = new SimulatedAnnealingStatsLogger(sa, loggingFreq);
		ConvergenceTrainerWithStatsLogging ct = new ConvergenceTrainerWithStatsLogging(sa, convergenceErrTol, maxIters, logger, iterTol);
		if(AlgosToRun[1]){
			ef.Reset();
			Long[] res = MeasureTime(ct);
			res[1] = ef.Count;
			times.put("sa", res);
			System.out.println("Done sa");
			results[1] = ct.Logger;
		}
		
		CrossoverFunction cf = new SingleCrossOver();
		MutationFunction mf = new DiscreteChangeOneMutation(ranges);
		GenericGeneticAlgorithmProblem ggap = new GenericGeneticAlgorithmProblem(ef, uniformDist, mf, cf);
		int populationSize = 100;
		int toMate = (int) (0.75 * populationSize);
		int toMutate = (int) (0.1 * populationSize);
		convergenceErrTol = 1e-5;
		StandardGeneticAlgorithm sga = new StandardGeneticAlgorithm(populationSize, toMate, toMutate, ggap);
		GeneticAlgorithmStatsLogger gaLogger = new GeneticAlgorithmStatsLogger(sga, loggingFreq);
		ConvergenceTrainerWithStatsLogging ggapTr = new ConvergenceTrainerWithStatsLogging(sga, convergenceErrTol, maxIters, gaLogger, iterTol);
		if(AlgosToRun[2]){
			ef.Reset();
			Long[] res = MeasureTime(ggapTr);
			res[1] = ef.Count;
			times.put("ga", res);
			System.out.println("Done ga");
			results[2] = ggapTr.Logger;
		}
		
		DiscreteDependencyTree ddt = new DiscreteDependencyTree(0.3, ranges);
		GenericProbabilisticOptimizationProblem gpop = new GenericProbabilisticOptimizationProblem(ef, uniformDist, ddt);
		int samples = 50;
		int tokeep = (int) (0.2 * samples); // this is the Nth percentile to keep inorder to recompute the new theta_(i+1)
		maxIters = 200000;
		iterTol = 50000;
		loggingFreq = 10;
		MIMIC mimic = new MIMIC(samples, tokeep, gpop);
		genericLogger = new GenericStatsLogger(loggingFreq);
		ConvergenceTrainerWithStatsLogging mimicTr = new ConvergenceTrainerWithStatsLogging(mimic, convergenceErrTol, maxIters, genericLogger, iterTol);
		if(AlgosToRun[3]){
			ef.Reset();
			Long[] res = MeasureTime(mimicTr);
			res[1] = ef.Count;
			times.put("mimic",res);
			System.out.println("Done mimic");
			results[3] = mimicTr.Logger;
		}
		
		return results;
	}
	
	public static StatsLogger[] GetResultsForCountOnesProblemWithSize(int size, int seed, HashMap<String, Long[]> times, boolean[] AlgosToRun){
		int N = size;
		int maxIters = 1200000;
		int iterTol = 1200000; // iters to wait before concluding convergence
		double convergenceErrTol = 1; // fn value should always increase by atleast this amount in each iter
		int loggingFreq = 10;
		StatsLogger[] results = new StatsLogger[4];
		Distribution.random.setSeed(seed);
		
		EvaluationFunction _ef = new CountOnesEvaluationFunction();
		EvaluationFunctionWithLogging ef = new EvaluationFunctionWithLogging(_ef);
		
		int[] ranges = GetArray(N, 2);
		DiscreteUniformDistribution uniformDist = new DiscreteUniformDistribution(ranges);
		NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
		HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, uniformDist, nf);
		GenericStatsLogger genericLogger = new GenericStatsLogger(loggingFreq);
		
		RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
		ConvergenceTrainerWithStatsLogging ctRhc = new ConvergenceTrainerWithStatsLogging(rhc, convergenceErrTol, maxIters, genericLogger, iterTol);
		if(AlgosToRun[0]){
			ef.Reset();
			Long[] res = MeasureTime(ctRhc);
			res[1] = ef.Count;
			times.put("rhc", res);
			System.out.println("Done rhc");
			results[0] = ctRhc.Logger;
		}
		
		double initialTemp = 1e11;
		double coolingRate = 0.99;
		SimulatedAnnealing sa = new SimulatedAnnealing(initialTemp, coolingRate, hcp);
		SimulatedAnnealingStatsLogger logger = new SimulatedAnnealingStatsLogger(sa, loggingFreq);
		ConvergenceTrainerWithStatsLogging ct = new ConvergenceTrainerWithStatsLogging(sa, convergenceErrTol, maxIters, logger, iterTol);
		if(AlgosToRun[1]){
			ef.Reset();
			Long[] res = MeasureTime(ct);
			res[1] = ef.Count;
			times.put("sa", res);
			System.out.println("Done sa");
			results[1] = ct.Logger;
		}
		
		CrossoverFunction cf = new SingleCrossOver();
		MutationFunction mf = new DiscreteChangeOneMutation(ranges);
		GenericGeneticAlgorithmProblem ggap = new GenericGeneticAlgorithmProblem(ef, uniformDist, mf, cf);
		int populationSize = 100;
		int toMate = (int) (0.25 * populationSize);
		int toMutate = (int) (0.1 * populationSize);
		convergenceErrTol = 1e-5;
		StandardGeneticAlgorithm sga = new StandardGeneticAlgorithm(populationSize, toMate, toMutate, ggap);
		GeneticAlgorithmStatsLogger gaLogger = new GeneticAlgorithmStatsLogger(sga, loggingFreq);
		ConvergenceTrainerWithStatsLogging ggapTr = new ConvergenceTrainerWithStatsLogging(sga, convergenceErrTol, maxIters, gaLogger, iterTol);
		if(AlgosToRun[2]){
			ef.Reset();
			Long[] res = MeasureTime(ggapTr);
			res[1] = ef.Count;
			times.put("ga", res);
			System.out.println("Done ga");
			results[2] = ggapTr.Logger;
		}
		
		DiscreteDependencyTree ddt = new DiscreteDependencyTree(0.1, ranges);
		GenericProbabilisticOptimizationProblem gpop = new GenericProbabilisticOptimizationProblem(ef, uniformDist, ddt);
		int samples = 50;
		int tokeep = (int) (0.2 * samples); // this is the Nth percentile to keep inorder to recompute the new theta_(i+1)
		maxIters = 200000;
		iterTol = 50000;
		loggingFreq = 1;
		MIMIC mimic = new MIMIC(samples, tokeep, gpop);
		genericLogger = new GenericStatsLogger(loggingFreq);
		MimicStatsLogger mimicLogger = new MimicStatsLogger(mimic, loggingFreq, ddt);
		ConvergenceTrainerWithStatsLogging mimicTr = new ConvergenceTrainerWithStatsLogging(mimic, convergenceErrTol, maxIters, mimicLogger, iterTol);
		if(AlgosToRun[3]){
			ef.Reset();
			Long[] res = MeasureTime(mimicTr);
			res[1] = ef.Count;
			times.put("mimic",res);
			System.out.println("Done mimic");
			results[3] = mimicTr.Logger;
			System.out.println("Dependency tree for size " + size);
			System.out.println(ddt.toString());
		}
		
		return results;
	}

	public static StatsLogger[] GetResultsForTwoColorsProblemWithSize(int size, int seed, HashMap<String, Long[]> times, boolean[] AlgosToRun){
		int N = size;
		int maxIters = 1200000;
		int iterTol = 1200000; // iters to wait before concluding convergence
		double convergenceErrTol = 1; // fn value should always increase by atleast this amount in each iter
		int loggingFreq = 10;
		StatsLogger[] results = new StatsLogger[4];
		Distribution.random.setSeed(seed);
		
		EvaluationFunction _ef = new TwoColorsEvaluationFunction();
		EvaluationFunctionWithLogging ef = new EvaluationFunctionWithLogging(_ef);
		
		int[] ranges = GetArray(N, 2);
		DiscreteUniformDistribution uniformDist = new DiscreteUniformDistribution(ranges);
		NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
		HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, uniformDist, nf);
		GenericStatsLogger genericLogger = new GenericStatsLogger(loggingFreq);
		
		RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
		ConvergenceTrainerWithStatsLogging ctRhc = new ConvergenceTrainerWithStatsLogging(rhc, convergenceErrTol, maxIters, genericLogger, iterTol);
		if(AlgosToRun[0]){
			ef.Reset();
			Long[] res = MeasureTime(ctRhc);
			res[1] = ef.Count;
			times.put("rhc", res);
			System.out.println("Done rhc");
			results[0] = ctRhc.Logger;
		}
		
		double initialTemp = 1e11;
		double coolingRate = 0.99;
		SimulatedAnnealing sa = new SimulatedAnnealing(initialTemp, coolingRate, hcp);
		SimulatedAnnealingStatsLogger logger = new SimulatedAnnealingStatsLogger(sa, loggingFreq);
		ConvergenceTrainerWithStatsLogging ct = new ConvergenceTrainerWithStatsLogging(sa, convergenceErrTol, maxIters, logger, iterTol);
		if(AlgosToRun[1]){
			ef.Reset();
			Long[] res = MeasureTime(ct);
			res[1] = ef.Count;
			times.put("sa", res);
			System.out.println("Done sa");
			results[1] = ct.Logger;
		}
		
		CrossoverFunction cf = new UniformCrossOver();
		MutationFunction mf = new DiscreteChangeOneMutation(ranges);
		GenericGeneticAlgorithmProblem ggap = new GenericGeneticAlgorithmProblem(ef, uniformDist, mf, cf);
		int populationSize = 200;
		int toMate = (int) (0.75 * populationSize);
		int toMutate = (int) (0.1 * populationSize);
		convergenceErrTol = 1e-5;
		StandardGeneticAlgorithm sga = new StandardGeneticAlgorithm(populationSize, toMate, toMutate, ggap);
		GeneticAlgorithmStatsLogger gaLogger = new GeneticAlgorithmStatsLogger(sga, loggingFreq);
		ConvergenceTrainerWithStatsLogging ggapTr = new ConvergenceTrainerWithStatsLogging(sga, convergenceErrTol, maxIters, gaLogger, iterTol);
		if(AlgosToRun[2]){
			ef.Reset();
			Long[] res = MeasureTime(ggapTr);
			res[1] = ef.Count;
			times.put("ga", res);
			System.out.println("Done ga");
			results[2] = ggapTr.Logger;
		}
		
		DiscreteDependencyTree ddt = new DiscreteDependencyTree(0.30, ranges);
		GenericProbabilisticOptimizationProblem gpop = new GenericProbabilisticOptimizationProblem(ef, uniformDist, ddt);
		int samples = 100;
		int tokeep = (int) (0.2 * samples); // this is the Nth percentile to keep inorder to recompute the new theta_(i+1)
		maxIters = 200000;
		iterTol = 50000;
		loggingFreq = 10;
		MIMIC mimic = new MIMIC(samples, tokeep, gpop);
		genericLogger = new GenericStatsLogger(loggingFreq);
		ConvergenceTrainerWithStatsLogging mimicTr = new ConvergenceTrainerWithStatsLogging(mimic, convergenceErrTol, maxIters, genericLogger, iterTol);
		if(AlgosToRun[3]){
			ef.Reset();
			Long[] res = MeasureTime(mimicTr);
			res[1] = ef.Count;
			times.put("mimic",res);
			System.out.println("Done mimic");
			results[3] = mimicTr.Logger;
		}
		
		return results;
	}
	
	public static StatsLogger[] GetResultsForKnapSackProblemWithSize(int size, int seed, HashMap<String, Long[]> times, boolean[] AlgosToRun){
		Random r = new Random(0);
		StatsLogger[] results = new StatsLogger[4];
		int N = size;
		int COPIES_EACH = 4;
		int MAX_WEIGHT = 50;
		int MAX_VOLUME = 50;
		double frac = 0.3 + (1 - 0.3) * r.nextDouble();
		int KNAPSACK_VOLUME = (int) (MAX_VOLUME * N * COPIES_EACH * frac);
		int[] copies = GetArray(N, COPIES_EACH);
		double[] weights = GetArray(N, 0.0);
		double[] volumes = GetArray(N, 0.0);
		
		for(int i = 0; i < N; i++){
			weights[i] = r.nextDouble() * MAX_WEIGHT;
			volumes[i] = r.nextDouble() * MAX_VOLUME;
		}
		
		int[] ranges = GetArray(N, COPIES_EACH + 1);
		
		int maxIters = 1200000;
		int iterTol = 120000; // iters to wait before concluding convergence
		double convergenceErrTol = 1; // fn value should always increase by atleast this amount in each iter
		int loggingFreq = 50;
		
		Distribution.random.setSeed(seed);
		EvaluationFunction _ef = new KnapsackEvaluationFunction(weights, volumes, KNAPSACK_VOLUME, copies);
		EvaluationFunctionWithLogging ef = new EvaluationFunctionWithLogging(_ef);
		
		DiscreteUniformDistribution uniformDist = new DiscreteUniformDistribution(ranges);
		NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
		HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, uniformDist, nf);
		GenericStatsLogger genericLogger = new GenericStatsLogger(loggingFreq);
		
		RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
		ConvergenceTrainerWithStatsLogging ctRhc = new ConvergenceTrainerWithStatsLogging(rhc, convergenceErrTol, maxIters, genericLogger, iterTol);
		if(AlgosToRun[0]){
			ef.Reset();
			Long[] res = MeasureTime(ctRhc);
			res[1] = ef.Count;
			times.put("rhc", res);

			System.out.println("Done rhc");
			results[0] = ctRhc.Logger;
			ef.Reset();
		}
		
		double initialTemp = 1e11;
		double coolingRate = 0.99;
		SimulatedAnnealing sa = new SimulatedAnnealing(initialTemp, coolingRate, hcp);
		SimulatedAnnealingStatsLogger logger = new SimulatedAnnealingStatsLogger(sa, loggingFreq);
		ConvergenceTrainerWithStatsLogging ct = new ConvergenceTrainerWithStatsLogging(sa, convergenceErrTol, maxIters, logger, iterTol);
		if(AlgosToRun[1]){
			ef.Reset();
			Long[] res = MeasureTime(ct);
			res[1] = ef.Count;
			times.put("sa", res);

			System.out.println("Done sa");
			results[1] = ct.Logger;
			ef.Reset();
		}
		
		CrossoverFunction cf = new SingleCrossOver();
		MutationFunction mf = new DiscreteChangeOneMutation(ranges);
		GenericGeneticAlgorithmProblem ggap = new GenericGeneticAlgorithmProblem(ef, uniformDist, mf, cf);
		int populationSize = 200;
		int toMate = (int) (0.75 * populationSize);
		int toMutate = (int) (0.1 * populationSize);
		convergenceErrTol = 1e-5;
		StandardGeneticAlgorithmWithTournamentSelection sga = new StandardGeneticAlgorithmWithTournamentSelection(populationSize, toMate, toMutate, ggap);
		GeneticAlgorithmStatsLogger gaLogger = new GeneticAlgorithmWithTournamentSelectionStatsLogger(sga, loggingFreq);
		ConvergenceTrainerWithStatsLogging ggapTr = new ConvergenceTrainerWithStatsLogging(sga, convergenceErrTol, maxIters, gaLogger, iterTol);
		if(AlgosToRun[2]){
			ef.Reset();
			Long[] res = MeasureTime(ggapTr);
			res[1] = ef.Count;
			times.put("ga", res);
			System.out.println("Done ga");
			results[2] = ggapTr.Logger;
		}
		
		DiscreteDependencyTree ddt = new DiscreteDependencyTree(0.2, ranges);
		GenericProbabilisticOptimizationProblem gpop = new GenericProbabilisticOptimizationProblem(ef, uniformDist, ddt);
		int samples = 100;
		int tokeep = (int) (0.2 * samples); // this is the Nth percentile to keep inorder to recompute the new theta_(i+1)
		maxIters = 200000;
		iterTol = 2000;
		MIMIC mimic = new MIMIC(samples, tokeep, gpop);
		genericLogger = new GenericStatsLogger(loggingFreq);
		ConvergenceTrainerWithStatsLogging mimicTr = new ConvergenceTrainerWithStatsLogging(mimic, convergenceErrTol, maxIters, genericLogger, iterTol);
		if(AlgosToRun[3]){
			ef.Reset();
			Long[] res = MeasureTime(mimicTr);
			res[1] = ef.Count;
			times.put("mimic",res);
			System.out.println("Done mimic");
			results[3] = mimicTr.Logger;
		}
		
		return results;
	}
	
	
	public static StatsLogger[] GetResultsForFlipFlopProblemWithSize(int size, int seed, HashMap<String, Long[]> times, boolean[] AlgosToRun){
		int N = size;
		int maxIters = 1200000;
		int iterTol = 1200000; // iters to wait before concluding convergence
		double convergenceErrTol = 1; // fn value should always increase by atleast this amount in each iter
		int loggingFreq = 10;
		StatsLogger[] results = new StatsLogger[4];
		
		Distribution.random.setSeed(seed);
		EvaluationFunction _ef = new FlipFlopEvaluationFunction();
		EvaluationFunctionWithLogging ef = new EvaluationFunctionWithLogging(_ef);
		
		int[] ranges = GetArray(N, 2);
		DiscreteUniformDistribution uniformDist = new DiscreteUniformDistribution(ranges);
		NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
		HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, uniformDist, nf);
		GenericStatsLogger genericLogger = new GenericStatsLogger(loggingFreq);
		
		RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
		ConvergenceTrainerWithStatsLogging ctRhc = new ConvergenceTrainerWithStatsLogging(rhc, convergenceErrTol, maxIters, genericLogger, iterTol);
		if(AlgosToRun[0]){
			ef.Reset();
			Long[] res = MeasureTime(ctRhc);
			res[1] = ef.Count;
			times.put("rhc", res);
			System.out.println("Done rhc");
			results[0] = ctRhc.Logger;
		}
		
		double initialTemp = 1e11;
		double coolingRate = 0.99;
		SimulatedAnnealing sa = new SimulatedAnnealing(initialTemp, coolingRate, hcp);
		SimulatedAnnealingStatsLogger logger = new SimulatedAnnealingStatsLogger(sa, loggingFreq);
		ConvergenceTrainerWithStatsLogging ct = new ConvergenceTrainerWithStatsLogging(sa, convergenceErrTol, maxIters, logger, iterTol);
		if(AlgosToRun[1]){
			ef.Reset();
			Long[] res = MeasureTime(ct);
			res[1] = ef.Count;
			times.put("sa", res);
			
			System.out.println("Done sa");
			results[1] = ct.Logger;
		}
		
		CrossoverFunction cf = new SingleCrossOver();
		MutationFunction mf = new DiscreteChangeOneMutation(ranges);
		GenericGeneticAlgorithmProblem ggap = new GenericGeneticAlgorithmProblem(ef, uniformDist, mf, cf);
		int populationSize = 200;
		int toMate = (int) (0.75 * populationSize);
		int toMutate = (int) (0.1 * populationSize);
		convergenceErrTol = 1e-5;
		StandardGeneticAlgorithm sga = new StandardGeneticAlgorithm(populationSize, toMate, toMutate, ggap);
		GeneticAlgorithmStatsLogger gaLogger = new GeneticAlgorithmStatsLogger(sga, loggingFreq);
		ConvergenceTrainerWithStatsLogging ggapTr = new ConvergenceTrainerWithStatsLogging(sga, convergenceErrTol, maxIters, gaLogger, iterTol);
		if(AlgosToRun[2]){
			ef.Reset();
			Long[] res = MeasureTime(ggapTr);
			res[1] = ef.Count;
			times.put("ga", res);
			
			System.out.println("Done ga");
			results[2] = ggapTr.Logger;
		}
		
		DiscreteDependencyTree ddt = new DiscreteDependencyTree(0.25, ranges);
		GenericProbabilisticOptimizationProblem gpop = new GenericProbabilisticOptimizationProblem(ef, uniformDist, ddt);
		int samples = 100;
		int tokeep = (int) (0.20 * samples); // this is the Nth percentile to keep inorder to recompute the new theta_(i+1)
		MIMIC mimic = new MIMIC(samples, tokeep, gpop);
		genericLogger = new GenericStatsLogger(loggingFreq);
		maxIters = 200000;
		iterTol = 50000;
		ConvergenceTrainerWithStatsLogging mimicTr = new ConvergenceTrainerWithStatsLogging(mimic, convergenceErrTol, maxIters, genericLogger, iterTol);
		if(AlgosToRun[3]){
			ef.Reset();
			Long[] res = MeasureTime(mimicTr);
			res[1] = ef.Count;
			times.put("mimic",res);
			System.out.println("Done mimic : " + mimic.getOptimizationProblem().value(mimic.getOptimal()) + " " + ((EvaluationFunctionWithLogging)ef).Count);
			results[3] = mimicTr.Logger;
		}
		
		return results;
	}

	public static StatsLogger[] GetResultsForTspProblemWithSize(int size, int seed, HashMap<String, Long[]> times, boolean[] AlgosToRun){
		int N = size;
		int maxIters = 1200000;
		int iterTol = 1200000; // iters to wait before concluding convergence
		double convergenceErrTol = 0.0001; // fn value should always increase by atleast this amount in each iter
		int loggingFreq = 10;
		StatsLogger[] results = new StatsLogger[4];
		
		Distribution.random.setSeed(seed);
		Random r = new Random(10);
		double[][] points = new double[N][2];
		for(int i = 0; i<N; i++){
			points[i][0] = r.nextDouble();
			points[i][1] = r.nextDouble();
		}
		EvaluationFunction _ef = new TravelingSalesmanRouteEvaluationFunction(points);
		EvaluationFunctionWithLogging ef = new EvaluationFunctionWithLogging(_ef);
		
		DiscretePermutationDistribution dpd = new DiscretePermutationDistribution(N);
		
		NeighborFunction nf = new SwapNeighbor();
		HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, dpd, nf);
		GenericStatsLogger genericLogger = new GenericStatsLogger(loggingFreq);
		
		RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
		ConvergenceTrainerWithStatsLogging ctRhc = new ConvergenceTrainerWithStatsLogging(rhc, convergenceErrTol, maxIters, genericLogger, iterTol);
		if(AlgosToRun[0]){
			ef.Reset();
			Long[] res = MeasureTime(ctRhc);
			res[1] = ef.Count;
			times.put("rhc", res);
			System.out.println("Done rhc");
			results[0] = ctRhc.Logger;
		}
		
		double initialTemp = 1e11;
		double coolingRate = 0.99;
		SimulatedAnnealing sa = new SimulatedAnnealing(initialTemp, coolingRate, hcp);
		SimulatedAnnealingStatsLogger logger = new SimulatedAnnealingStatsLogger(sa, loggingFreq);
		ConvergenceTrainerWithStatsLogging ct = new ConvergenceTrainerWithStatsLogging(sa, convergenceErrTol, maxIters, logger, iterTol);
		if(AlgosToRun[1]){
			ef.Reset();
			Long[] res = MeasureTime(ct);
			res[1] = ef.Count;
			times.put("sa", res);
			System.out.println("Done sa");
			results[1] = ct.Logger;
		}
		
		CrossoverFunction cf = new TravelingSalesmanCrossOver((TravelingSalesmanRouteEvaluationFunction)_ef);
		MutationFunction mf = new SwapMutation();
		GenericGeneticAlgorithmProblem ggap = new GenericGeneticAlgorithmProblem(ef, dpd, mf, cf);
		int populationSize = 200;
		int toMate = (int) (0.75 * populationSize);
		int toMutate = (int) (0.1 * populationSize);
		convergenceErrTol = 1e-5;
		StandardGeneticAlgorithm sga = new StandardGeneticAlgorithm(populationSize, toMate, toMutate, ggap);
		GeneticAlgorithmStatsLogger gaLogger = new GeneticAlgorithmStatsLogger(sga, loggingFreq);
		ConvergenceTrainerWithStatsLogging ggapTr = new ConvergenceTrainerWithStatsLogging(sga, convergenceErrTol, maxIters, gaLogger, iterTol);
		if(AlgosToRun[2]){
			ef.Reset();
			Long[] res = MeasureTime(ggapTr);
			res[1] = ef.Count;
			times.put("ga", res);
			System.out.println("Done ga");
			results[2] = ggapTr.Logger;
		}
		
		int[] ranges = GetArray(N, N);
		_ef = new TravelingSalesmanSortEvaluationFunction(points);
		DiscreteUniformDistribution uniformDist = new DiscreteUniformDistribution(ranges);
		DiscreteDependencyTree ddt = new DiscreteDependencyTree(0.1, ranges);
		GenericProbabilisticOptimizationProblem gpop = new GenericProbabilisticOptimizationProblem(_ef, uniformDist, ddt);
		int samples = 100;
		int tokeep = (int) (0.2 * samples); // this is the Nth percentile to keep inorder to recompute the new theta_(i+1)
		maxIters = 200000;
		iterTol = 50000;
		loggingFreq = 10;
		MIMIC mimic = new MIMIC(samples, tokeep, gpop);
		genericLogger = new GenericStatsLogger(loggingFreq);
		ConvergenceTrainerWithStatsLogging mimicTr = new ConvergenceTrainerWithStatsLogging(mimic, convergenceErrTol, maxIters, genericLogger, iterTol);
		if(AlgosToRun[3]){
			ef.Reset();
			Long[] res = MeasureTime(mimicTr);
			res[1] = ef.Count;
			times.put("mimic",res);
			System.out.println("Done mimic");
			results[3] = mimicTr.Logger;
		}
		
		return results;
	}

	
	public static int[] GetArray(int size, int defaultValue){
		int a[] = new int[size];
		for(int i = 0; i < size; i++){
			a[i] = defaultValue;
		}
		return a;
	}
	
	public static double[] GetArray(int size, double defaultValue){
		double a[] = new double[size];
		for(int i = 0; i < size; i++){
			a[i] = defaultValue;
		}
		return a;
	}

	public static Long[] MeasureTime(Trainer trainer){
		long startTime = System.currentTimeMillis();
		double error = trainer.train();
		System.out.println(error);
		return new Long[]{(System.currentTimeMillis() - startTime) / 1000,(long) 0};
	}
	
	public static long MeasureTimeMilliSec(Trainer trainer){
		long startTime = System.currentTimeMillis();
		trainer.train();
		return (System.currentTimeMillis() - startTime);
	}
	
	public static void CreateDirectoryIfNotExists(String path){
		File dir = new File(path);
		dir.mkdirs();
	}
}
