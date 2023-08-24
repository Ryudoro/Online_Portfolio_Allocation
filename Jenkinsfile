pipeline {
    agent any
    
    stages {
        stage('Checkout') {
            steps {
                git 'https://github.com/Ryudoro/Online_Portfolio_Allocation.git'
            }
        }
        
        stage('Install Dependencies') {
            steps {
                sh 'pip install -r requirements.txt'
            }
        }
        
        stage('Data Preparation and Model Training') {
            steps {
                sh 'python Model/input_creation.py'
                // sh 'python model_training.py'
            }
        }
        
        // stage('Predictions and Visualization') {
        //     steps {
        //         sh 'python make_predictions.py'
        //         sh 'python visualize_results.py'
        //     }
        // }
        
        stage('Tests') {
            steps {
                sh 'python -m unittest discover tests'
            }
        }
        
        // stage('Deploy (Optional)') {
        //     when {
        //         branch 'master'
        //     }
        //     steps {
        //         sh 'python deploy.py'
        //     }
        // }
    }
    
    post {
        always {
            archiveArtifacts artifacts: '**/*', allowEmptyArchive: true
        }
    }
}

