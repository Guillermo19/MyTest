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
	private Dataset<Row> test;

	private String featuresFile;
	private String testFile;
	private String[] columnNames;
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
		testFile = in.nextLine();
		file = new File(testFile);
		while(!file.exists()){
			System.out.println("Enter the name of the file you want to check for possible errors:: ");
			testFile = in.nextLine();
			file = new File(testFile);
		}
		in.close();
		
		features = sparkSession.read()
				  .option("header", "true")
				  .option("inferSchema", "true")
				  .option("delimiter", "\t")
				  .csv(featuresFile);
		
		test = sparkSession.read()
				  .option("header", "true")
				  .option("inferSchema", "true")
				  .option("delimiter", "\t")
				  .csv(testFile);
		
		features.cache();
		test.cache();
		columnNames = test.columns();
		featureColumns = features.columns();
	}
	
	private void labelAnomalies(){
		
	}
	//prepares the dataset for features on the specified columns 
	private void prepareDataset(){
		VectorAssembler featureAssembler;
		List<String> featureCols = new ArrayList<>();
		StringIndexerModel indexer;
		
		for(String column : columnNames){
			if(features.head().getAs(column).getClass().getSimpleName().equals("String")){
				Column col = new Column(column);
				
				indexer = new StringIndexer().setInputCol(column).setOutputCol("INDEX"+column).fit(features);
				features = indexer.transform(features);
				
				featureCols.add("INDEX"+column);
			}
			else
				featureCols.add(column);
		}

		String[] features = featureCols.toArray(new String[0]);
		featureAssembler = new VectorAssembler().setInputCols(features).setOutputCol("features");
		
		features = featureAssembler.transform(features);
		test = featureAssembler.transform(test);
		
	}
	
	//checks for anomalous data features the model on the whole dataset
	private void checkForAnomalies(){
		this.prepareDataset();
		classifier = new GBTClassifier().setLabelCol("anomalous");
		model = classifier.fit(features);
	}
	
	//displaus the possible anomalies to the user  
	public void printAnomalies(){
		this.checkForAnomalies();
		Column anomaly = new Column("prediction");
		Dataset<Row> anomalies = model.transform(test).filter(anomaly.equalTo(1));
		System.out.println("Possible anomalous data: ");
		anomalies.select("prediction", columnNames).drop("prediction").show();
		System.out.println(model.toDebugString());
		//System.out.println(model);
	}
}
