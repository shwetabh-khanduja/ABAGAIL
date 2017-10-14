package assignment2;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.HashMap;

import func.nn.NetworkTrainer;
import func.nn.activation.LogisticSigmoid;
import func.nn.backprop.BackPropagationNetwork;
import func.nn.backprop.BackPropagationNetworkFactory;
import func.nn.backprop.StandardUpdateRule;
import func.nn.backprop.StochasticBackPropagationTrainer;
import opt.OptimizationAlgorithm;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.NeuralNetworkOptimizationProblem;
import opt.ga.StandardGeneticAlgorithm;
import shared.DataSet;
import shared.ErrorMeasure;
import shared.Instance;
import shared.SumOfSquaresError;
import shared.Trainer;
import shared.reader.CSVDataSetReader;
import shared.writer.CSVWriter;
import util.linalg.Vector;

public class NeuralNetworkExperiments {
	
	public static class TrainResults{
		public ArrayList<Double> Loss;
		public ArrayList<int[]> TrainOutput;
		public ArrayList<int[]> ValidationOutput;
		public ArrayList<int[]> TestOutput;
		public long Time;
		public BackPropagationNetwork Network;
		public Metrics TrainMetrics;
		public Metrics ValidationMetrics;
		public Metrics TestMetrics;
		
		public void PrintResult(){
			System.out.println("Train Metrics : " + TrainMetrics.ToString());
			System.out.println("Validation Metrics : " + ValidationMetrics.ToString());
			System.out.println("Test Metrics : " + TestMetrics.ToString());
			System.out.println("Time : " + Time);
			System.out.println("Iters : " + Loss.size());
		}
	}
	
	public static class Parameters{
		public String TrainDataFile;
		public String ValidationDataFile;
		public String TestDataFile;
		public int InputNodes;
		public int HiddenNodes;
		public int OutputNodes;
		public double InitialTemp = 1e11;
		public double CoolingRate = 0.95;
		public int PopulationSize = 100;
		public double MatingFrac = 0.20;
		public double MutateFrac = 0.01;
		public double LearningRate = 0.1;
		public double Momentum = 0.9;
		public double L2Imp = 0.001;
		public int[] Iters = new int[]{2000,2000,2000,2000};
		public boolean[] AlgosToRun;
	}
	
	public static HashMap<String, TrainResults> RunNeuralNetwork(Parameters parameters) throws Exception{
		HashMap<String, TrainResults> output = new HashMap<>();
		
		DataSet train = LoadDataSet(parameters.TrainDataFile,parameters);
		DataSet validation = LoadDataSet(parameters.ValidationDataFile, parameters);
		DataSet test = LoadDataSet(parameters.TestDataFile, parameters);
		
		parameters.InputNodes = train.get(0).size();
		
		ErrorMeasure error = new SumOfSquaresError();
		BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
		BackPropagationNetwork[] networks = new BackPropagationNetwork[4];
		NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[4];
		OptimizationAlgorithm[] oa = new OptimizationAlgorithm[4];
		String[] names = new String[]{"rhc","sa","ga","bp"};
		
		for(int i = 0; i< names.length; i++){
			networks[i] = factory.createClassificationNetwork(
					new int[]{parameters.InputNodes,parameters.HiddenNodes,parameters.OutputNodes},
					new LogisticSigmoid());	
			nnop[i] = new NeuralNetworkOptimizationProblem(train, networks[i], error);
		}
		if(parameters.AlgosToRun[0]) oa[0] = new RandomizedHillClimbing(nnop[0]);
		
		if(parameters.AlgosToRun[1]) oa[1] = new SimulatedAnnealing(parameters.InitialTemp,parameters.CoolingRate,nnop[1]);
		
		if(parameters.AlgosToRun[2])
			oa[2] = new StandardGeneticAlgorithm(parameters.PopulationSize, 
					(int)(parameters.PopulationSize * parameters.MatingFrac), 
					(int)(parameters.PopulationSize * parameters.MutateFrac), 
					nnop[2]);
		
		ArrayList<Double> loss = new ArrayList<>();
		for(int i = 0; i < 3; i++){
			if(!parameters.AlgosToRun[i]) continue;
			loss = new ArrayList<>();
			long time = Train(oa[i],networks[i],train,error,parameters.Iters[i],loss);
			TrainResults tr = new NeuralNetworkExperiments.TrainResults();
			tr.Time = time;
			tr.TrainOutput = GetPredictedAndActualLabels(networks[i], train);
			tr.ValidationOutput = GetPredictedAndActualLabels(networks[i], validation);
			tr.TestOutput = GetPredictedAndActualLabels(networks[i], test);
			tr.Network = networks[i];
			tr.Loss = loss;
			tr.TrainMetrics = new Metrics(tr.TrainOutput);
			tr.ValidationMetrics = new Metrics(tr.ValidationOutput);
			tr.TestMetrics = new Metrics(tr.TestOutput);
			output.put(names[i], tr);
			PrintLineWithTimestamp("done " + names[i]);
			tr.PrintResult();
		}
		
		// now for the nn using backpropagation
		if(parameters.AlgosToRun[3]){
			loss = new ArrayList<>();
			NetworkTrainer bpTrainer = new StochasticBackPropagationTrainer(
					train, networks[3], new SumOfSquaresError(), 
					new StandardUpdateRule(parameters.LearningRate, parameters.Momentum));
			
//			NetworkTrainer bpTrainer = new StochasticBackPropagationTrainer(
//			train, networks[3], new SumOfSquaresError(), 
//			new L2PenaltyWeightUpdateRule(parameters.LearningRate, parameters.Momentum, parameters.L2Imp));
			
			ConvergenceTrainerWithStatsLogging trainer = new ConvergenceTrainerWithStatsLogging
					(bpTrainer, 1e-5, parameters.Iters[3], new GenericStatsLogger(1), 10);
			long time = Train(trainer,networks[3],train,error,1,loss);
			TrainResults tr = new NeuralNetworkExperiments.TrainResults();
			tr.Time = time;
			tr.TrainOutput = GetPredictedAndActualLabels(networks[3], train);
			tr.ValidationOutput = GetPredictedAndActualLabels(networks[3], validation);
			tr.TestOutput = GetPredictedAndActualLabels(networks[3], test);
			tr.Network = networks[3];
			tr.Loss = ((GenericStatsLogger)trainer.Logger).Loss;
			tr.TrainMetrics = new Metrics(tr.TrainOutput);
			tr.ValidationMetrics = new Metrics(tr.ValidationOutput);
			tr.TestMetrics= new Metrics(tr.TestOutput);
			output.put(names[3], tr);
			PrintLineWithTimestamp("done " + names[3]);
			tr.PrintResult();
		}
		return output;
	}
	
	public static long Train(Trainer oa, BackPropagationNetwork network, DataSet train, ErrorMeasure measure, int Iters, ArrayList<Double> loss){
		long time = 0;
		double tol = 1e-2;
		int tolIter = 10;
		double lastError = 0.0;
		int printFreq = 500;
		for(int i=0; i < Iters; i++){
			time += Runs.MeasureTimeMilliSec(oa);
			double error = ComputeLoss(network, train, measure);
			loss.add(error);
			if(printFreq == 0){
				System.out.println(i + " " + error);
				printFreq = 500;
			}
			--printFreq;
			if(Math.abs(lastError - error) < tol){
				--tolIter;
			}
			else{
				tolIter = 500;
			}
			if(tolIter == 0){
				System.out.println("converged at iter : " + i);
				break;
			}
		}
		return time;
	}
	
	public static Double ComputeLoss(BackPropagationNetwork network, DataSet train, ErrorMeasure measure){
		double error = 0;
		for (Instance instance : train) {
			network.setInputValues(instance.getData());
			network.run();
			Instance output = instance.getLabel();
			Vector output_values = network.getOutputValues();
			Instance example = new Instance(output_values, new Instance(output_values.get(0)));
			error += measure.value(output, example);
		}
		return error;
	}
	
	public static ArrayList<int[]> GetPredictedAndActualLabels(BackPropagationNetwork network, DataSet data){
		ArrayList<int[]> ret = new ArrayList<>();
		for (Instance instance : data) {
			network.setInputValues(instance.getData());
			network.run();
			double predicted = network.getOutputValues().get(0);
			double actual = instance.getLabel().getContinuous();
			int predictedLabel = (int) (Math.abs(predicted-actual) < 0.5 ? actual : Math.abs(1-actual));
			ret.add(new int[]{predictedLabel,(int) actual});
		}
		return ret;
	}
	
	public static DataSet LoadDataSet(String dataFile, Parameters parameters) throws Exception{
		CSVDataSetReader reader = new CSVDataSetReader(dataFile);
		DataSet data = reader.read();
		String labelFile = dataFile.replace(".csv", "_label.csv");
		DataSet labels = new CSVDataSetReader(labelFile).read();
		for(int i = 0; i < labels.size(); i++){
			Instance label = labels.get(i);
			data.get(i).setLabel(label);
		}
		return data;
	}

	public static String GetCurrentTime(){
		String timeStamp = new SimpleDateFormat("yyyy-MM-dd_HH-mm-ss").format(Calendar.getInstance().getTime());
		return timeStamp;
	}
	
	public static void PrintLineWithTimestamp(String line){
		System.out.println("["+GetCurrentTime()+"]:"+line);
	}
	
	public static String ConvertToString(String del, ArrayList<Double> values){
		String[] strs = new String[values.size()];
		for(int i = 0; i < values.size() ; i++){
			strs[i] = values.get(i).toString();
		}
		return String.join(del, strs);
	}
	
	public static void Perform(String rootFolder, String outputFileName, String sizes) throws Exception{
		String[] datasetSizes = sizes.split(",");
		String outputFile = rootFolder + "/nnets/" + outputFileName;
		CSVWriter writer = new CSVWriter(outputFile, "size,algo,time,train_acc,valid_acc,test_acc,train_f1,valid_f1,test_f1,loss".split(","));
		writer.open();
		String[] algos = new String[]{"rhc","sa","ga","bp"};
		for (String size : datasetSizes) {
			PrintLineWithTimestamp("starting dataset " + size);
			String folder = rootFolder + "/VowelRecognition/" + size;
			Parameters p = new Parameters();
			p.TrainDataFile = folder + "/train.csv";
			p.Momentum = 0.9;
			p.ValidationDataFile = folder + "/validation.csv";
			p.TestDataFile = folder + "/test.csv";
			p.HiddenNodes = 30;
			p.LearningRate = 0.1;
			p.OutputNodes = 1;
			p.Iters = new int[]{10000,10000,10000,10000};
			p.L2Imp = 0.0001;
			p.MutateFrac = 0.01;
			p.MatingFrac = 0.20;
			p.AlgosToRun = new boolean[]{true,true,true,true};
			HashMap<String, TrainResults> tr = RunNeuralNetwork(p);
			for(int i =0; i < algos.length; i++){
				if(tr.containsKey(algos[i])){
					TrainResults results = tr.get(algos[i]);
					writer.write(size + "," + algos[i] + "," + results.Time + "," + results.TrainMetrics.Acc + "," + results.ValidationMetrics.Acc + "," + results.TestMetrics.Acc + "," + results.TrainMetrics.F1 + "," + results.ValidationMetrics.F1+ "," + results.TestMetrics.F1 + "," + ConvertToString(";",results.Loss));
					writer.nextRecord();
				}
			}
		}
		writer.close();
	}
}
