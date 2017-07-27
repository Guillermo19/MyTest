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
		
		training = training.withColumn("anomalous", functions.lit(0));
		columnNames = test.columns();
	}
	
	//elements on the columnName column that are not equal to the limit object are labeled as bad on the training set
	public void setSingleLimit(String columnName, Object limit){
		Column column = new Column(columnName);
		training = training.withColumn("anomalous", functions.when((column.notEqual(limit)).or((anomalous).equalTo(1)), 1).otherwise(0));
	}
	
	//elements on the columnName column that are lower than the lowerLimit object are labeled as bad on the training set
	public void setLowerLimit(String columnName, Object lowerLimit){
		Column column = new Column(columnName);
		training = training.withColumn("anomalous", functions.when((column.lt(lowerLimit)).or((anomalous).equalTo(1)), 1).otherwise(0));
	}
	
	//elements on the columnName column that are greater than the upperLimit object are labeled as bad on the training set
	public void setUpperLimit(String columnName, Object upperLimit){
		Column column = new Column(columnName);
		training = training.withColumn("anomalous", functions.when((column.gt(upperLimit)).or((anomalous).equalTo(1)), 1).otherwise(0));
	}
	
	//prepares the dataset for training on the specified columns 
	private void prepareDataset(String ... columns){
		StringIndexerModel indexer;
		VectorAssembler assembler;
		List<String> featureCols = new ArrayList<>();
		String[] features;
		
		for(String column : columns){
			if(training.first().getAs(column).getClass().getSimpleName().equals("String")){
				
				indexer = new StringIndexer().setInputCol(column).setOutputCol("index"+column).fit(training);
				
				training = indexer.transform(training);
				test = indexer.transform(test);
				
				featureCols.add("index"+column);
			}
			
			else
				featureCols.add(column);
		}
		
		features = featureCols.toArray(new String[0]);
		
		assembler = new VectorAssembler().setInputCols(features).setOutputCol("features");
		
		training = assembler.transform(training);
		test = assembler.transform(test);

	}
	
	//check for anomalous data training the model on the specified columns
	private void checkForAnomalies(String ... columns){
		this.prepareDataset(columns);
		classifier = new GBTClassifier().setLabelCol("anomalous");
		model = classifier.fit(training);
	}
	
	//checks for anomalous data training the model on the whole dataset
	private void checkForAnomalies(){
		this.prepareDataset(training.columns());
		classifier = new GBTClassifier().setLabelCol("anomalous");
		model = classifier.fit(training);

	}
	
	public void printAnomalies(String ... columns){
		this.checkForAnomalies(columns);
		Column anomaly = new Column("prediction").equalTo(1);
		Dataset<Row> anomalies = model.transform(test).filter(anomaly);
		anomalies.select("prediction", columnNames).show();
	}
	
	public void printAnomalies(){
		this.checkForAnomalies();
		Column anomaly = new Column("prediction");
		Dataset<Row> anomalies = model.transform(test).filter(anomaly.equalTo(1));
		anomalies.select("prediction", columnNames).show();
	}
}
