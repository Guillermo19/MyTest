import java.util.List;
import java.util.Scanner;
import java.io.File;
import java.io.IOException;
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
	private Dataset<Row> trainingSet;

	private String featuresOrModelFile;
	private String datasetFile;
	private String modelNametoSave;
	
	private List<String> featureColumns;
	
	private GBTClassifier classifier;
	private GBTClassificationModel model;
	
	public DetectAnomalies() {
		File file;
		Scanner in = new Scanner(System.in);
		System.out.println("Enter the name of the file you want to check for possible errors: ");
		datasetFile = in.nextLine();
		file = new File(datasetFile);
		while(!file.exists()){
			System.out.println("Enter the name of the file you want to check for possible errors: ");
			datasetFile = in.nextLine();
			file = new File(datasetFile);
		}
		dataset = sparkSession.read()
				  .option("header", "true")
				  .option("inferSchema", "true")
				  .option("delimiter", "\t")
				  .csv(datasetFile);
	
		dataset.cache();
		System.out.println("Enter the model you want to use for prediction OR enter the features file if you want to create a new model: ");
		featuresOrModelFile = in.nextLine();
		file = new File(featuresOrModelFile);
		while(!file.exists()){
			System.out.println("Enter the model you want to use for prediction OR enter the features file if you want to create a new model: ");
			featuresOrModelFile = in.nextLine();
			file = new File(featuresOrModelFile);
		}
		
		try{
			model = GBTClassificationModel.load(featuresOrModelFile);
			//this.prepareDataset();
		}catch(Exception e){
			e.printStackTrace();
			features = sparkSession.read()
					  .text(featuresOrModelFile);
			
			features.cache();
			
			System.out.println("No model detected... Creating new model based on features entered.");
			System.out.println("Enter the name of the model to save: ");
			modelNametoSave = in.nextLine();
			dataset = dataset.withColumn("label", functions.lit(0));
			this.labelAnomalies();
			this.prepareDataset();
			this.createModel();
		}finally{
			in.close();
		}
	}
	
	private void labelAnomalies(){
		List<Row> conditions = features.collectAsList();
		String[] columns = dataset.columns();
		featureColumns = new ArrayList<>();
		
		for(Row row : conditions){
			String condition = row.getString(0);
			Dataset<Row> temp =  dataset.filter(condition);
			if(temp.count() > 0){
				dataset = dataset.except(temp);
				temp = temp.withColumn("label", functions.lit(1));
				dataset = dataset.union(temp);
			}
			
			for(String col : columns){
				if(condition.contains(col.toUpperCase())){
					featureColumns.add(col);
				}
			}
		}
		
		System.out.println("finished1");
		trainingSet = dataset;
		trainingSet.cache();
	}
	
	//prepares the dataset for features on the specified columns 
	private void prepareDataset(){
		VectorAssembler featureAssembler;
		List<String> featureCols = new ArrayList<>();
		List<String> columnsToCheck = new ArrayList<>();
		
		StringIndexerModel indexer;
		
		for(String column : featureColumns){
			if(trainingSet.head().getAs(column).getClass().getSimpleName().equals("String")){
				indexer = new StringIndexer().setInputCol(column).setOutputCol("INDEX"+column).fit(trainingSet);
				trainingSet = indexer.transform(trainingSet);
				featureCols.add("INDEX"+column);
			}
			else if(trainingSet.head().getAs(column).getClass().getSimpleName().equals("Timestamp")){
				trainingSet = trainingSet.withColumn("INDEX"+column, new Column(column).cast("Integer"));
				featureCols.add("INDEX"+column);
			}
			else
				featureCols.add(column);
			
			columnsToCheck.add(column);
		}
		
		String[] features = featureCols.toArray(new String[0]);
		featureAssembler = new VectorAssembler().setInputCols(features).setOutputCol("features");
		trainingSet = featureAssembler.transform(trainingSet);
		
		try {
			new VectorAssembler()
			.setInputCols(columnsToCheck.toArray(new String[0]))
			.setOutputCol("features")
			.save(modelNametoSave+"assembler");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		System.out.println("finished2");
	}
	
	//checks for anomalous data features the model on the whole dataset
	private void createModel(){
		System.out.println("start");
		classifier = new GBTClassifier();
		model = classifier.fit(trainingSet);
		System.out.println("middle");
		try {
			model.save(modelNametoSave);
			System.out.println("Model saved!");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	//displaus the possible anomalies to the user  
	public void printAnomalies(){
		Column anomaly = new Column("prediction");
		Dataset<Row> anomalies = model.transform(dataset).filter(anomaly.equalTo(1));
		System.out.println("Possible anomalous data: ");
		anomalies.select("prediction", featureColumns.toArray(new String[0])).drop("prediction").show(50);
		//System.out.println(model.toDebugString());
	}
}
