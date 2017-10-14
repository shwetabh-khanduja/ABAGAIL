package assignment2;

import java.io.Console;
import java.util.Arrays;
import java.util.HashMap;

import opt.EvaluationFunction;
import shared.Instance;
import util.linalg.Vector;

public class EvaluationFunctionWithLogging implements EvaluationFunction{

		public long Count;
		public EvaluationFunction Func;
		public HashMap<String, Double> Cache;
		public double CacheHit = 0;
		StringBuilder sb = new StringBuilder();
		
		public EvaluationFunctionWithLogging(EvaluationFunction f) {
			this.Func = f;
			Cache = new HashMap<>();
		}

		@Override
		public double value(Instance d) {
			++Count;
			return Func.value(d);
//			String key = GetKey(d);
//			double value = Cache.getOrDefault(key, -9999.0);
//			if(value != -9999){
//				CacheHit++;
////				System.out.println(CacheHit + " " + Count + " " + CacheHit/Count);
//				return value;
//			}
//			++Count;
//			value = Func.value(d);
//			Cache.put(key, value);
//			return value;
		}
		
		public String GetKey(Instance d){
			sb.setLength(0);
			Vector v = d.getData();
			for(int i = 0; i < v.size(); i++){
				sb.append(v.get(i));
			}
			return sb.toString();
		}
		
		public void Reset(){
			System.out.println("Before resetting : " + Count + " " + CacheHit);
			Count = 0;
			CacheHit = 0;
			Cache.clear();
		}
}
