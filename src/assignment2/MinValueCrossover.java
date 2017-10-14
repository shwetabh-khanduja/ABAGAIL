package assignment2;

import dist.Distribution;
import opt.ga.CrossoverFunction;
import shared.Instance;

public class MinValueCrossover implements CrossoverFunction{
	public Instance mate(Instance a, Instance b) {
        // Create space for the mated solution
        double[] newData = new double[a.size()];

        // Assign bits to the mated solution
        for (int i = 0; i < newData.length; i++) {
            double aVal = a.getContinuous(i);
            double bVal = b.getContinuous(i);
            double newVal = aVal > bVal ? aVal : bVal;
            newData[i] = newVal;
        }

        // Return the mated solution
        return new Instance(newData);
    }
}
