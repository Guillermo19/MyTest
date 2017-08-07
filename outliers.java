import java.util.ArrayList;
import java.util.List;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.clustering.GaussianMixture;
import org.apache.spark.ml.clustering.GaussianMixtureModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class OutlierDetection {
	private GaussianMixture mixture; 
	private GaussianMixtureModel model;
	
	private Dataset<Row> dataset;
	private Dataset<Row> anomalies;
	
	private String[] columns;
	private final static double threshold = 0.2;
	
	private final static SparkConf sparkConf = 
			new SparkConf()
			.setAppName("Outlier Detection")
			.setMaster("local");
	
	private final static JavaSparkContext sparkContext = new JavaSparkContext(sparkConf);
	
	private final static SparkSession sparkSession = 
			SparkSession
			  .builder()
			  .appName("Outlier Detection")
			  .getOrCreate();
	
	public OutlierDetection(String filePath) {
	
		dataset = sparkSession.read().
				option("header", "true").
				option("delimiter", "\t").
				option("inferSchema", "true").
				csv(filePath);  //reads the data from a file whose path is passed to the constructor 
		
		dataset.cache();
		
		columns = dataset.columns();
		
		List<String> featureColumns = new ArrayList<>();
		
		for(String column : dataset.columns()){
			String className = dataset.select(column).head().getAs(0).getClass().getSimpleName();
			
			if(className.equals("Integer") || className.equals("Double")){
				featureColumns.add(column);
			}
		}
		
		featureColumns.remove("ID");
		
		VectorAssembler assembler = new VectorAssembler()
				.setInputCols(featureColumns.toArray(new String[0]))
				.setOutputCol("features");
		
		dataset = assembler.transform(dataset);
		
		mixture = new GaussianMixture();
		model = mixture.fit(dataset);

	} //end constructor
	
	//identifies potential outliers that might be mixed with normal data
	private void getAnomalies(){ 
		long count = dataset.count();
		long[] clusterSizes = model.summary().clusterSizes();
		int possibleAnomalous = 0;
		
		while(possibleAnomalous != -1){
			possibleAnomalous = -1;
			for(int i = 0; i<clusterSizes.length; i++){
				if((double) (clusterSizes[i])/count <= threshold){
					possibleAnomalous = i;
				}
			}
			
			if(possibleAnomalous != -1){
				Column prediction = new Column("prediction");
				dataset = model.transform(dataset);
				anomalies = dataset.filter(prediction.equalTo(possibleAnomalous));
			}
		}

	}//end get anomalies
	
	public void printAnomalies() {
		this.getAnomalies();
		if(anomalies.count() > 0)
			anomalies.select("prediction", columns).drop("prediction").show();
		else
			System.out.println("No anomalies to report");
		
	} //end print anomalies 
}//end class
