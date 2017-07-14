import java.util.ArrayList;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.clustering.BisectingKMeans;
import org.apache.spark.ml.clustering.BisectingKMeansModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;

public class anomalies {
	private LinearRegression linReg;
	private LinearRegressionModel model;
	
	private Dataset<Row> dataset;
	private Dataset<Row> vectorDataset;
	private Dataset<Row> residuals;
	
	private ArrayList<Row> anomalies;
	
	private final static double threshold = 2.0;
	
	private final static SparkConf sparkConf = 
			new SparkConf()
			.setAppName("Anomaly Detection")
			.setMaster("local");
	
	private final static JavaSparkContext sparkContext = new JavaSparkContext(sparkConf);
	
	private final static SparkSession sparkSession = 
			SparkSession
			  .builder()
			  .appName("Anomaly Detection")
			  .getOrCreate();
	
	public anomalies(String filePath) {
	
		dataset = sparkSession.read().
				option("header", "true").
				option("delimiter", "\t").
				option("inferSchema", "true").
				csv(filePath);  //reads the data from a file whose path is passed to the constructor 
		
		dataset.cache(); 
		
		VectorAssembler assembler = new VectorAssembler()
				.setInputCols(new String[]{"Size"})
				.setOutputCol("features");
		
		vectorDataset = assembler.transform(dataset); //converts and combines the 'size' and 'records' column as the vector type 'features' column
		
		linReg = new LinearRegression().setLabelCol("Volume");
		
		model = linReg.fit(vectorDataset); //trains the k-means with the dataset to create a model
		
	} //end constructor
	
	private void areThereAnomalies(){
		residuals = model.summary().residuals();
		double residualStddev = sparkContext.parallelizeDoubles(residuals.toJavaRDD().map(f -> f.getDouble(0)).collect()).stdev();
		
	}
	private void getAnomalies(){ //identifies potential outliers that might be mixed with normal data and saves them
		
	}//end get anomalies
	
	public void printAnomalies() {
		this.areThereAnomalies();
		//vectorDataset.show(25,false);
		/*if(!anomalies.isEmpty()){
			System.out.println("Possible anomalous files: ");
			for(Row r : anomalies){
				System.out.println(r.getAs("File").toString());
			}
		}
		else{
			System.out.println("No probable anomalous files");
		}*/
		
	} //end print anomalies 
}
