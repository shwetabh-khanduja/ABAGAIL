package assignment2;

import func.nn.backprop.BackPropagationLink;
import func.nn.backprop.WeightUpdateRule;

public class L2PenaltyWeightUpdateRule extends WeightUpdateRule{
    /**
     * The learning rate to use
     */
    private double learningRate;
    
    /**
     * The momentum to use
     */
    private double momentum;
    
    private double penalty;

    /**
     * Create a new standard momentum update rule
     * @param learningRate the learning rate
     * @param momentum the momentum
     */
    public L2PenaltyWeightUpdateRule(double learningRate, double momentum, double penalty) {
        this.momentum = momentum;
        this.learningRate = learningRate;  
        this.penalty = penalty;
    }
    
    /**
     * Create a new standard update rule
     */
    public L2PenaltyWeightUpdateRule() {
    	this(.2, .9, 0.0001);
    }

    /**
     * @see nn.backprop.BackPropagationUpdateRule#update(nn.backprop.BackPropagationLink)
     */
    public void update(BackPropagationLink link) {
    	double currentWeight = link.getWeight();
    	double l2Delta = (1 - 2 * learningRate * penalty);
    	double newWeight = currentWeight * l2Delta - 
    			learningRate * link.getError() + 
    			link.getLastChange() * momentum;
        link.setWeight(newWeight);
    }
}
