package assignment2;

import func.nn.NeuralNetwork;
import shared.Instance;
import shared.SumOfSquaresError;

public class SumOfSquaresdErrorWithL2WeightsPenalty extends SumOfSquaresError {
	
	private NeuralNetwork network;
	private double imp;
	
	public SumOfSquaresdErrorWithL2WeightsPenalty(NeuralNetwork network, double imp){
		this.network = network;
		this.imp = imp;
	}
	
	@Override
	public double value(Instance output, Instance example) {
        double wtSum = 0.0;
        double[] wts = network.getWeights();
        for(int i = 0; i < wts.length; i++){
        	wtSum += wts[i] * wts[i];	
        }
        return super.value(output, example) + imp * wtSum;
    }
	
	@Override
	public double[] gradient(Instance output, Instance example) {      
        double[] errorArray = new double[output.size()];
        Instance label = example.getLabel();
        for (int i = 0; i < output.size(); i++) {
            errorArray[i] = (output.getContinuous(i) - label.getContinuous(i))
                * example.getWeight() + 2 * imp * example.getWeight();
        }
        return errorArray;
    }
}
