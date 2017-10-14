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
		
		public EvaluationFunctionWithLogging(EvaluationFunction f) {
			this.Func = f;
		}

		@Override
		public double value(Instance d) {
			++Count;
			return Func.value(d);
		}
		
		public void Reset(){
			Count = 0;
		}
}
