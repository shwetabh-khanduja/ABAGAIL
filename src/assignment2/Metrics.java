package assignment2;

import java.util.ArrayList;

public class Metrics {
	public double TP;
	public double FP;
	public double FN;
	public double TN;
	public double F1;
	public double Acc;
	public double P;
	public double N;
	
	public Metrics(ArrayList<int[]> PredAct){
		TP = 0.0;
		FP = 0.0; 
		FN = 0.0;
		TN = 0.0;
		for(int i = 0; i < PredAct.size(); i++){
			int pred = PredAct.get(i)[0];
			int act = PredAct.get(i)[1];
			if(pred == 1 && act == 1){
				TP++;
			}
			else if(pred == 1 && act == 0){
				FP++;
			}
			else if(pred == 0 && act == 1){
				FN++;
			}
			else{ // pred == 0 && act == 0
				TN++;
			}
		}
		
		Acc = (TP + TN) / (TP + TN + FP + FN);
		F1 = 2 * TP / (2 * TP + FP + FN);
	}
	
	public String ToString(){
		return "Acc:" + Acc + " F1:"+ F1 + " TP:" + TP + " FP:" + FP + " FN:" + FN + " TN:" + TN;
	}
}
