package assignment2;

import java.util.ArrayList;

import opt.SimulatedAnnealing;


public class SimulatedAnnealingStatsLogger extends GenericStatsLogger {

	public SimulatedAnnealing sa;
	public ArrayList<Double> Temperatures;
		
	public SimulatedAnnealingStatsLogger(SimulatedAnnealing sa, int freq) {
		super(freq);
		this.sa = sa;
		Temperatures = new ArrayList<>();
	}
	
	@Override
	public boolean log(int iter, double loss, boolean force) {
		if(super.log(iter, loss, force)){
			Temperatures.add(sa.currentTemp());
			return true;
		}
		return false;
	}
}
