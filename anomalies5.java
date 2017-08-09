import java.util.List;
import java.util.Scanner;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.classification.GBTClassificationModel;
import org.apache.spark.ml.classification.GBTClassifier;
import org.apache.spark.ml.feature.StringIndexer;
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
			featureColumns = Arrays.asList(VectorAssembler.load(featuresOrModelFile + "assembler").getInputCols());
			this.prepareDataset();
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
		String bigCondition = "";
		featureColumns = new ArrayList<>();
		int i = 0;
		
		for(Row row : conditions){
			String condition = row.getString(0);
			for(String col : columns){
				if(condition.contains(col.toUpperCase())){
					featureColumns.add(col);
				}
			}
			
			if(i > 0)
				bigCondition = bigCondition + " OR (" + condition + ")";
			else
				bigCondition = condition;
			
			i++;
		}
		
		Dataset<Row> temp = dataset.filter(bigCondition);
		if(temp.count() > 0){
			dataset = dataset.except(temp);
			temp = temp.withColumn("label", functions.lit(1));
			dataset = dataset.union(temp);
		}
		
		dataset.cache();
	}
	
	//prepares the dataset for features on the specified columns 
	private void prepareDataset(){
		VectorAssembler featureAssembler;
		StringIndexer indexer = new StringIndexer();
		
		String[] features = new String[featureColumns.size()];
		int i = 0;
		
		for(String column : featureColumns){
			if(dataset.head().getAs(column).getClass().getSimpleName().equals("String")){
				dataset = indexer
						.setInputCol(column)
						.setOutputCol("INDEX"+column)
						.fit(dataset)
						.transform(dataset);
				
				features[i++] = "INDEX" + column;
			}
			else if(dataset.head().getAs(column).getClass().getSimpleName().equals("Timestamp")){
				dataset = dataset.withColumn("INDEX"+column, new Column(column).cast("Integer"));
				features[i++] = "INDEX" + column;
			}
			else
				features[i++] = column;
		}
		
		featureAssembler = new VectorAssembler().setInputCols(features).setOutputCol("features");
		dataset = featureAssembler.transform(dataset);
		
		try {
			new VectorAssembler()
			.setInputCols(featureColumns.toArray(new String[0]))
			.save(modelNametoSave+"assembler");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	//checks for anomalous data features the model on the whole dataset
	private void createModel(){
		System.out.println("start");
		classifier = new GBTClassifier();
		model = classifier.fit(dataset);
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
		anomalies.select("prediction", featureColumns.toArray(new String[0])).drop("prediction").show();
		//System.out.println(model.toDebugString());
	}
}
