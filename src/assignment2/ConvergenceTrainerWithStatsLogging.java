package assignment2;

import shared.ConvergenceTrainer;
import shared.Trainer;

public class ConvergenceTrainerWithStatsLogging extends ConvergenceTrainer{
	
	public StatsLogger Logger;
	public int IterTol;
	
	public ConvergenceTrainerWithStatsLogging(Trainer trainer, double threshold, 
			int maxIterations, StatsLogger logger, int IterTol) {
		super(trainer, threshold, maxIterations);
		this.Logger = logger;
		this.IterTol = IterTol;
	}
	
	@Override
	public double train() {
		double lastError;
        double error = Double.MAX_VALUE;
        int _iterTol = IterTol;
        boolean lastLogged = false;
        do {
           iterations++;
           lastError = error;
           error = trainer.train();
           lastLogged =  Logger.log(iterations, error, false);
           if(Math.abs(error - lastError) <= threshold){
        	   --_iterTol;
           }
           else{
        	   _iterTol = IterTol;
           }
        } while (_iterTol > 0 && iterations < maxIterations);
        if(!lastLogged){
        	Logger.log(iterations, error, true);
        }
        System.out.println(error);
        return error;
	}
}
