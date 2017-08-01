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
	private Dataset<Row> unlabeledData;
	
	private String trainingFile;
	private String testFile;
	private String[] columnNames;
	
	private GBTClassifier classifier;
	private GBTClassificationModel model;
	
	public DetectAnomalousData() {
		File file;
		Scanner in = new Scanner(System.in);
		System.out.println("Enter the name of the file you want to train the model with: ");
		trainingFile = in.nextLine();
		file = new File(trainingFile);
		while(!file.exists()){
			System.out.println("Enter the name of the file you want to train the model with: ");
			trainingFile = in.nextLine();
			file = new File(trainingFile);
		}
		
		System.out.println("Enter the name of the file you want to check for possible erros: ");
		testFile = in.nextLine();
		file = new File(testFile);
		while(!file.exists()){
			System.out.println("Enter the name of the file you want to train the model with: ");
			testFile = in.nextLine();
			file = new File(testFile);
		}
		in.close();
		
		training = sparkSession.read()
				  .option("header", "true")
				  .option("inferSchema", "true")
				  .option("delimiter", "\t")
				  .csv(trainingFile);
		
		test = sparkSession.read()
				  .option("header", "true")
				  .option("inferSchema", "true")
				  .option("delimiter", "\t")
				  .csv(testFile);
		
		training.cache();
		test.cache();
		columnNames = test.columns();
	}
	
	//prepares the dataset for training on the specified columns 
	private void prepareDataset(){
		VectorAssembler featureAssembler;
		List<String> featureCols = new ArrayList<>();
		StringIndexerModel indexer;
		Dataset<Row> temp = test;
		
		for(String column : columnNames){
			if(training.head().getAs(column).getClass().getSimpleName().equals("String")){
				Column col = new Column(column);
				
				indexer = new StringIndexer().setInputCol(column).setOutputCol("INDEX"+column).fit(training);
				training = indexer.transform(training);
				
				List<String> knownValues = training.select(col).toJavaRDD().map(f -> f.getString(0)).collect();
				this.checkUnknownData(temp, column, knownValues);
				test = indexer.transform(test.filter(col.isin(knownValues.toArray())));
				
				featureCols.add("INDEX"+column);
			}
			else
				featureCols.add(column);
		}

		String[] features = featureCols.toArray(new String[0]);
		featureAssembler = new VectorAssembler().setInputCols(features).setOutputCol("features");
		
		training = featureAssembler.transform(training);
		test = featureAssembler.transform(test);
		
	}
	
	//checks for anomalous data training the model on the whole dataset
	private void checkForAnomalies(){
		this.prepareDataset();
		classifier = new GBTClassifier().setLabelCol("anomalous");
		model = classifier.fit(training);
	}
	
	//displaus the possible anomalies to the user  
	public void printAnomalies(){
		this.checkForAnomalies();
		Column anomaly = new Column("prediction");
		Dataset<Row> anomalies = model.transform(test).filter(anomaly.equalTo(1));
		System.out.println("Possible anomalous data: ");
		anomalies.select("prediction", columnNames).drop("prediction").show();
		//System.out.println(model.toDebugString());
	}
	
	//checks for categorical values NOT present on the training dataset and removes them from the testing set
	private void checkUnknownData(Dataset<Row> temp, String column, List<String> knownValues){
		Column col = new Column(column);
		Column condition = temp.col(column).equalTo(temp.filter(col.isin(knownValues.toArray())).col(column));
		unlabeledData = temp.join(temp.filter(col.isin(knownValues.toArray())), condition, "leftanti");
		if(unlabeledData.count() > 0){
			System.out.println("\nUnknown value(s) in: " + testFile);
			unlabeledData.select(col).show();
		}
	}
}
