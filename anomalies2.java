import java.util.List;
import java.util.ArrayList;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.classification.GBTClassificationModel;
import org.apache.spark.ml.classification.GBTClassifier;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;

public class DetectAnomalousData {
	
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
	
	private Dataset<Row> training;
	private Dataset<Row> test;
	
	private final Column anomalous = new Column("anomalous");
	private String[] columnNames;
	
	private GBTClassifier classifier;
	private GBTClassificationModel model;
	
	public DetectAnomalousData(String goodFile, String badFile) {
	
		training = sparkSession.read()
				  .option("header", "true")
				  .option("inferSchema", "true")
				  .option("delimiter", "\t")
				  .csv(goodFile);
		
		test = sparkSession.read()
				  .option("header", "true")
				  .option("inferSchema", "true")
				  .option("delimiter", "\t")
				  .csv(badFile);
		
		training.cache();
		test.cache();
		columnNames = test.columns();
	}
	
	//prepares the dataset for training on the specified columns 
	private void prepareDataset(){
		VectorAssembler assembler;
		
		for(String col : training.columns()) {
			if(training.head().getAs(col).getClass().getSimpleName().equals("String")){
				List<Row> stringValues = training.select(col).distinct().collectAsList();
				
				for(Row row : stringValues){
					Object val = row.get(0);
					training = training.withColumn("HASH"+col, functions.when(new Column(col).equalTo(val).and(other), val.hashCode()));
				}
			}
		}
		
		
		assembler = new VectorAssembler().setInputCols(training.columns()).setOutputCol("features");
		
		training = assembler.transform(training);
		test = assembler.transform(test);

	}
	
	//checks for anomalous data training the model on the whole dataset
	private void checkForAnomalies(){
		this.prepareDataset();
		classifier = new GBTClassifier().setLabelCol("anomalous");
		model = classifier.fit(training);

	}
	
	public void printAnomalies(){
		this.checkForAnomalies();
		Column anomaly = new Column("prediction");
		Dataset<Row> anomalies = model.transform(test).filter(anomaly.equalTo(1));
		anomalies.select("prediction", columnNames).show();
	}
}
