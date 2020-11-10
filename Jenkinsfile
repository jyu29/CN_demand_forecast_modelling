pipeline {
    agent any
    environment {
        branch = 'develop'
        gitUrl = 'https://github.com/dktunited/forecast-modeling-demand/'
    }
    stages {
        stage('Demand Forecast Modeling') {
            agent {
              label 'PYLINUXT3XLARGE'
            }
            
            environment {
                conda_env = 'forecast-modeling-demand'
            }
            steps {
                git changelog: false, credentialsId: 'github_dktjenkins', poll: false, url: "${gitUrl}", branch: "${branch}"
                sh('''
                source ~/miniconda3/etc/profile.d/conda.sh
                conda deactivate
                conda remove --name ${conda_env} --all
                conda env create -f environment.yml
                conda activate ${conda_env}
                python main.py --environment ${run_env} --list_cutoff ${list_cutoff}
                conda deactivate
                // conda remove --name ${conda_env} --all
                ''')
            }
        }
    }

    post {
            failure {
                    mail to: 'benjamin.bouillet@decathlon.com',
                    subject: "Pipeline ${env.JOB_NAME} failed", body: "${env.BUILD_URL}"
                    }

            unstable {
                    mail to: 'benjamin.bouillet@decathlon.com',
                    subject: "Pipeline ${env.JOB_NAME} unstable", body: "${env.BUILD_URL}"
                    }
        }
}

