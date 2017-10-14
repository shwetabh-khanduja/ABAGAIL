package assignment2;

import java.util.ArrayList;

public class GenericStatsLogger implements StatsLogger {
	public ArrayList<Double> Loss;
	public ArrayList<Integer> Iters;
	public int LastIteration;
	public int LoggingFreq; // will log value after these many calls
	private int _currentCount = 0;
	private boolean _firstLog = true;
	
	public GenericStatsLogger(int loggingFrequency) {
		Loss = new ArrayList<>();
		Iters = new ArrayList<>();
		LoggingFreq = loggingFrequency;
	}
	
	@Override
	public boolean log(int iter, double loss, boolean force) {
		if(ShouldLog() || force){
			Loss.add(loss);
			Iters.add(iter);
			LastIteration = iter;
			return true;
		}
		return false;
	}
	
	public boolean ShouldLog(){
		++_currentCount;
		if(_currentCount == LoggingFreq || _firstLog){
			if(_firstLog){
				_firstLog = false;
				if(LoggingFreq == 1){
					_currentCount = 0;
				}
			}
			else{
				_currentCount = 0;	
			}
			return true;
		}
		return false;
	}

	@Override
	public int Count() {
		return Loss.size();
	}
}
