package assignment2;

import shared.FixedIterationTrainer;
import shared.Trainer;

public class FixedIterationTrainerWithStatsLogging extends FixedIterationTrainer{
	public StatsLogger Logger;
	public FixedIterationTrainerWithStatsLogging(Trainer t, int iter, StatsLogger logger) {
		super(t, iter);
		this.Logger = logger;
	}
	@Override
	public double train(){
		double sum = 0;
        for (int i = 0; i < iterations; i++) {
        	double e = trainer.train();
            sum += e;
            Logger.log(i, e, false);
        }
        return sum / iterations;
	}
}
