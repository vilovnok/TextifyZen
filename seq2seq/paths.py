import os


ROOT = ''
RU_ADAPT_PATHS = [
     os.path.join(ROOT, 'RuAdapt', 'Adapted_literature', 'zlatoust_sentence_aligned_with_CATS.csv'),
     os.path.join(ROOT, 'RuAdapt', 'Encyclopedic', 'lsslovar_B_to_A_sent.csv'),
     os.path.join(ROOT, 'RuAdapt', 'Encyclopedic', 'lsslovar_C_to_A_sent.csv'),
     os.path.join(ROOT, 'RuAdapt', 'Encyclopedic', 'lsslovar_C_to_B_sent.csv'),
     os.path.join(ROOT, 'RuAdapt', 'Fairytales', 'df_fairytales_sent.csv')]
RUSIMPLESENTEVAL_PATH = os.path.join(ROOT, 'RuSimpleSentEval','dev_sents.csv')
TEST_DATA_PATH = os.path.join(ROOT, 'RuSimpleSentEval','public_test_sents.csv')
LOG_PATH = os.path.join(ROOT, 'train.logs')
CHECKPOINTS_FOLDER = os.path.join(ROOT, 'checkpoints')


def get_checkpoints_path(model_id, dataset):
     file_name = os.path.join(CHECKPOINTS_FOLDER, f'{model_id}-{dataset}.pt')
     if os.path.exists(CHECKPOINTS_FOLDER):
          if os.path.isfile(file_name):
               return os.path.join(CHECKPOINTS_FOLDER, f'{model_id}-{dataset}.pt')
          else:
               with open(file_name, 'w') as file:
                    pass 
               return os.path.join(CHECKPOINTS_FOLDER, f'{model_id}-{dataset}.pt')
     else:
          os.makedirs(CHECKPOINTS_FOLDER)     
          with open(file_name, 'w') as file:
               pass
          return os.path.join(CHECKPOINTS_FOLDER, f'{model_id}-{dataset}.pt')