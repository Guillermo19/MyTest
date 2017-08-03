import java.util.List;
import java.util.Scanner;
import java.io.File;
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

public class DetectAnomalies {
	
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
	
	private Dataset<Row> features;
	private Dataset<Row> dataset;

	private String featuresFile;
	private String datasetFile;
	private String[] featureColumns;
	
	private GBTClassifier classifier;
	private GBTClassificationModel model;
	
	public DetectAnomalies() {
		File file;
		Scanner in = new Scanner(System.in);
		System.out.println("Enter the features file: ");
		featuresFile = in.nextLine();
		file = new File(featuresFile);
		while(!file.exists()){
			System.out.println("Enter the features file: ");
			featuresFile = in.nextLine();
			file = new File(featuresFile);
		}
		
		System.out.println("Enter the name of the file you want to check for possible errors: ");
		datasetFile = in.nextLine();
		file = new File(datasetFile);
		while(!file.exists()){
			System.out.println("Enter the name of the file you want to check for possible errors:: ");
			datasetFile = in.nextLine();
			file = new File(datasetFile);
		}
		in.close();
		
		features = sparkSession.read()
				  .option("header", "true")
				  .option("inferSchema", "true")
				  .option("delimiter", "\t")
				  .csv(featuresFile);
		
		dataset = sparkSession.read()
				  .option("header", "true")
				  .option("inferSchema", "true")
				  .option("delimiter", "\t")
				  .csv(datasetFile);
		
		features.cache();
		dataset.cache();
		featureColumns = features.columns();
		dataset = dataset.withColumn("label", functions.lit(0));
	}
	
	private void labelAnomalies(){
		for(String column : featureColumns){
			Column col = new Column(column);
			Column label = new Column("label");
			String feature = features.head().getAs(column);
			Column condition = this.getCondition(feature, col);
			dataset = dataset.withColumn("label", functions.when(condition.or(label.equalTo(1)), 1).otherwise(0));
		}
	}
	
	private Column getCondition(String feature, Column col){
		Column condition = col.isNull();
		char filter = feature.charAt(0);
		Object value = feature.substring(1);
		
		switch(filter){
		case '=':
			condition = col.equalTo(value);
			break;
		case '!':
			condition = col.notEqual(value);
			break;
		case '>':
			condition = col.gt(value);
			break;
		case '<':
			condition = col.lt(value);
			break;
		case '%':
			condition = col.like("%"+value+"%");
			break;
		}
		
		return condition;
	}
	//prepares the dataset for features on the specified columns 
	private void prepareDataset(){
		this.labelAnomalies();
		VectorAssembler featureAssembler;
		List<String> featureCols = new ArrayList<>();
		StringIndexerModel indexer;
		
		for(String column : featureColumns){
			if(dataset.head().getAs(column).getClass().getSimpleName().equals("String")){
				
				indexer = new StringIndexer().setInputCol(column).setOutputCol("INDEX"+column).fit(dataset);
				dataset = indexer.transform(dataset);
				featureCols.add("INDEX"+column);
			}
			else
				featureCols.add(column);
		}

		String[] features = featureCols.toArray(new String[0]);
		featureAssembler = new VectorAssembler().setInputCols(features).setOutputCol("features");
		
		//features = featureAssembler.transform(features);
		dataset = featureAssembler.transform(dataset);
		
	}
	
	//checks for anomalous data features the model on the whole dataset
	private void checkForAnomalies(){
		this.prepareDataset();
		classifier = new GBTClassifier();
		model = classifier.fit(dataset);
	}
	
	//displaus the possible anomalies to the user  
	public void printAnomalies(){
		this.checkForAnomalies();
		Column anomaly = new Column("prediction");
		Dataset<Row> anomalies = model.transform(dataset).filter(anomaly.equalTo(1));
		System.out.println("Possible anomalous data: ");
		anomalies.select("prediction", featureColumns).drop("prediction").show();
		//System.out.println(model.toDebugString());
		//System.out.println(model);
	}
}
