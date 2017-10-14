package assignment2;

public interface StatsLogger {
	boolean log(int iter, double loss, boolean force);
	int Count();
}
