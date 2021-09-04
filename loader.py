#encoding:utf-8
from collections import  Counter
import tensorflow.keras as kr
import numpy as np
import codecs
import re
import jieba


def read_file(filename):
    """
    Args:
        filename:trian_filename,test_filename,val_filename
    Returns:
        two list where the first is lables and the second is contents cut by jieba

    """
    re_han = re.compile(u"([\u4E00-\u9FD5a-zA-Z]+)")  # the method of cutting text by punctuation
    contents, labels = [], []
    with codecs.open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                line = line.rstrip()
                # assert len(line.split('\t'))==2
                _,label,content = line.split('\t')
                labels.append(label)
                blocks = re_han.split(content)
                word = []
                for blk in blocks:
                    if re_han.match(blk):
                        for w in jieba.cut(blk):
                            if len(w) >= 2:
                                word.append(w)
                contents.append(word)
            except:
                pass
    return labels, contents

def build_vocab(filenames,vocab_dir,vocab_size=8000):
    """
    Args:
        filename:trian_filename,test_filename,val_filename
        vocab_dir:path of vocab_filename
        vocab_size:number of vocabulary
    Returns:
        writting vocab to vocab_filename

    """
    all_data = []
    for filename in filenames:
        _,data_train=read_file(filename)
        for content in data_train:
            all_data.extend(content)
    print(str(all_data))
    counter=Counter(all_data)
    count_pairs=counter.most_common(vocab_size-1)
    words,_=list(zip(*count_pairs))
    words=['<PAD>']+list(words)

    with codecs.open(vocab_dir,'w',encoding='utf-8') as f:
        f.write('\n'.join(words)+'\n')

def read_vocab(vocab_dir):
    """
    Args:
        filename:path of vocab_filename
    Returns:
        words: a list of vocab
        word_to_id: a dict of word to id
        
    """
    words=codecs.open(vocab_dir,'r',encoding='utf-8').read().strip().split('\n')
    word_to_id=dict(zip(words,range(len(words))))
    return words,word_to_id

def read_category():
    """
    Args:
        None
    Returns:
        categories: a list of label
        cat_to_id: a dict of label to id

    """
    categories = ['Unable to walk', 'Menstrual cramp', 'Chronic back pain', 'Depressed mood', 'Dysphagia', 'Wheezing', 'Immunosuppression', 'Difficulty dressing', 'Frequent night waking', 'Diplopia', 'Flatulence, eructation and gas pain', 'Hypertonia', 'Collapse', 'morphine', 'Aphasia', 'Uterus problem', 'ibuprofen', 'Visual disturbance', 'Heart valve regurgitation', 'Discolouration of skin', 'Epistaxis', 'Syncope', 'Lipex', 'Craving for food or drink', 'Arthritis of hand', 'Stumbling gait', 'Productive cough', 'Intercostal myalgia', 'Peeling of skin', 'Reduced concentration', 'Night pain', 'Parkinsonism', 'Erectile dysfunction', 'Lacks confidence', "Sj鰃ren's syndrome", 'Shoulder pain', 'Otalgia', 'Deep pain', 'Feeling content', 'Abdominal muscles tense', 'Pain in cheek', 'Speech impairment', 'Acute pain', 'Disorder of nasal sinus', 'Increased albumin', 'Menopause present', 'Lumbar facet joint pain', 'Slurred speech', 'Feeling nervous', 'Finding of increased blood pressure', 'Elbow joint pain', 'Panadeine', 'Low back strain', 'Feeling agitated 232 249', 'Hand eczema', 'Decreased venous pressure', 'Gastric ulcer', 'Nasal congestion', 'Renal failure', 'melatonin', 'Pain of uterus', 'Limping', 'Photosensitisation due to sun', 'Muscle strain', 'Disorder of foot', 'Lip swelling', 'oestradiol', 'Osteoarthritis', 'Swelling of hand', 'ezetimibe + simvastatin', 'Unable to move', 'Abdominal distension', 'Pain in left arm', 'Hand joint pain', 'Enbrel', 'Decreased range of shoulder movement', 'Pain in upper limb', 'Cut of hand', 'Hallucinations', 'Myalgia/myositis - multiple', 'Cold foot', 'Toprol-XL', 'Disturbance in speech rhythm', 'Shoulder tendonitis', 'Rupture tendon thigh', 'Symptom of ankle', 'Random blood sugar raised', 'B Complex (Cenovis)', 'Difficulty walking up stairs', 'Prolonged periods', 'Cervical arthritis', 'Celebrex 200 mg capsule: hard', 'Menorrhagia', 'Low back pain', 'Stiffness', 'Stomach cramps', 'Lipitor 40 mg tablet: film-coated', 'Tight chest', 'Diastolic dysfunction', 'lisinopril', 'Excessive and frequent menstruation', 'Zoloft', 'Contracture of tendon of ankle region', 'Rectal haemorrhage', 'Atacand', 'fenofibrate 145 mg tablet', 'Hearing loss', 'Tendonitis', 'Cramping pain', 'Abnormal gait', 'Severe vertigo', 'Drinks wine', 'Ankle oedema', 'Gland symptom', 'Rupture of tendon of lower leg and ankle', 'Generalised aches and pains', 'Unable to think clearly', 'Palpitations - rapid', 'Burning epigastric pain', 'Urgent desire to urinate', 'Abnormal weight gain', 'Tagamet', 'Foot swelling', 'Lightheadedness', 'Yellow skin', 'Clumsiness', 'Seroquel', 'Unable to find words', 'Gout', 'Walking disability', 'Reduced mobility', 'Oedema', 'Diovan', 'Impaired mobility', 'Frequent defaecation', 'Disorder of back', 'Arthritis of shoulder region joint', 'Ill-at-ease', 'Pulse fast', 'Reduced libido', 'Shin splint', 'Lupus erythematosus', 'Depression', 'Increased pressure', 'Mild cognitive disorder', 'Severe pain', 'Bad circulation - vasomotor change', 'General unsteadiness', 'Difficulty riding a bicycle', 'Generalised acute body pains', 'Dementia', 'C-reactive protein abnormal', 'Xanax', 'Dry skin', 'Fever', 'Pain in wrist', 'Allergic condition', 'Moderate pain', 'Male climacteric', 'Foot eczema', 'Hyperaesthesia', 'Impairment level of vision', 'Urinary frequency', 'Pravachol', 'Mentally dull', 'diclofenac', 'Lasix', 'Maxalt', 'Aptyalism', 'Uncontrollable vomiting', 'Neuropathy of upper limb', 'Discomfort', 'Andrews Tums Antacid', 'Compression neuropathy of trunk', 'Difficulty moving arm', 'Hypercholesterolaemia', 'Deep bruising', 'Periorbital oedema', 'Heavy legs', 'Parvovirus infection', 'Cannot sleep at all', 'Bone pain', 'Stabbing pain', 'Irregular heart rate', 'Pale complexion', 'Swelling of wrist joint', 'Bipolar disorder', 'Impaired cognition', 'Fatigue', 'Thoracic back pain', 'Disorder of extremity', 'dextropropoxyphene + paracetamol', 'Elevated cholesterol/high density lipoprotein ratio', 'Weight gain', 'Disorder of nervous system', 'Zantac', 'Eye pain', 'Stress overload', 'Burping', 'Disorder of tendon', 'Forgetful', 'Myalgia/myositis -pelvis/thigh', 'Coma', 'Reduced visual acuity', 'Traumatic injury of skeletal muscle', 'Swallowing problem', 'Numbness of toe', 'Muscle spasms of head AND/OR neck', 'Lateral epicondylitis', 'Post traumatic osteoarthritis', 'Clammy skin', 'Sensation of heaviness in limbs', 'Oedema of extremity', 'High pain threshold', 'Swollen nasal mucosa', 'Inflammatory disorder', 'Oxycontin', 'Unable to stand', 'Tear of meniscus of knee', 'Rheumatoid arthritis', 'Knee joint inflamed', 'Stiff limbs', 'Impairment of balance', 'Feeling intoxicated', 'Cramp', 'gemfibrozil', 'paracetamol', 'Dyspnoea', 'Polyneuropathy', 'Taste sense altered', 'Knee stiff', 'Blocked ears', 'Fosamax', 'Swelling of upper limb', 'Headache', 'Heart failure', 'Congestion of nasal sinus', 'Voltaren Rapid', 'Muscle fasciculation', 'Tendonitis of knee', 'Subdural haematoma', 'Severe dehydration', 'Non-smoker', 'Chill', 'Unable to lift', 'Warm hands', 'Pain provoked by movement', 'Upper respiratory tract infection', 'Disorientated', 'Deseril', 'Sleep deprivation', 'Cramp in calf', 'testosterone', 'Sight deteriorating', 'Numbness of face', 'Polymenorrhoea', 'Unable to stand up', 'vitamin A', 'Cramp in lower leg', 'Unable to speak', 'Weakness of limb', 'Swollen legs', 'Influenza', 'Serum creatinine raised', 'Crying associated with mood', 'Pain provoked by walking', 'Abnormal vision', 'misoprostol', 'Urinary retention', 'Pneumonia', 'Senility', 'Arthropathy', 'Degeneration of intervertebral disc', 'Lack of energy', 'Oedema of finger', 'Phlebitis', 'Erythema', 'Toe swelling', 'Diverticulitis', 'Cold sweat', 'Anxiety', 'Malaise and fatigue', 'Voltaren Rapid 25 mg tablet: sugar-coated', 'Irregular heart beat', 'Calf muscle weakness', 'Zestril', 'rabeprazole', 'Nasal discharge', 'Personality change', 'Disorder of cardiac function', 'Ringing in ear', 'Chronic pain', 'Feels everything is futile', 'Feeling agitated', 'Neuroma of foot', 'Propensity to adverse reactions to drug', 'Absent minded', 'Tinnitus', 'Voltaren Emulgel', 'Knee joint painful on movement', 'Malignant neoplastic disease', 'Isolated memory skills', 'Excessive upper gastrointestinal gas', 'Unable to move hand', 'Burning sensation', 'Infection', 'Serum triglycerides raised', 'Tendonitis AND/OR tenosynovitis of the elbow region', 'Neurontin', 'Hoarse', 'Paraesthesia of hand', 'Numbness of hand', 'paracetamol + codeine', 'Tongue swelling', 'Clicking interphalangeal joint of toe', 'Reduction of bulk of muscle', 'Arthropathy of knee joint', 'Vibration sensation present', 'Endometriosis of uterus', 'Ulcerative colitis', 'Spasmodic movement', 'Amyotrophic lateral sclerosis', 'Swelling of lower limb', 'Myocardial infarction', 'Heartburn', 'Difficulty sleeping', 'Orchidodynia', 'Scab of skin', 'omega-3-acid ethyl esters-90', 'Respiratory tract congestion', 'Emotional upset', 'Difficulty gripping', 'Dental caries', 'Repetitive strain injury', 'Postmenopausal bleeding', 'Impaired exercise tolerance', 'Weakness of hand', 'Achilles tendonitis', 'Abnormal dreams', 'Disorder of face', 'Has delayed recall', 'Numbness and tingling sensation of skin', 'Degenerative disorder of muscle', 'Injury of nervous system', 'Intolerant of cold', 'meloxicam', 'Metabolic syndrome X', 'Tremor', 'Weight loss', 'Acid reflux', 'Loose stool', 'Chronic fatigue syndrome', 'atorvastatin', 'Pain in spine', 'Malignant tumour of spleen', 'Distention of blood vessel', 'Difficulty controlling anger', 'Neuroma', 'Aggression', 'Contusion', 'Rash', 'Decreased muscle tone', 'Capoten', 'metoprolol', 'Premenstrual tension syndrome', 'Increased appetite', 'bisoprolol', 'tramadol', 'Numbness of limbs', 'Numbness', 'Unable to eat', 'Pain provoked by rest', 'Skin nodule', 'Swollen feet', 'Prinivil', 'Swelling of knee joint', 'Cramp in lower limb', 'Unable to walk down stairs', 'Myalgia/myositis - lower leg', 'Feeling tired', 'Haemorrhage of abdominal cavity structure', 'Myalgia/myositis - shoulder', 'Right upper quadrant pain', 'Immobile', 'Tenderness of upper limb', 'Abnormal large bowel motility', 'Derangement of meniscus', 'Bunion', 'Incoordination', 'Abdominal colic', 'ubidecarenone', 'Hypothyroidism', 'Renal pain', 'Rupture of tendon', 'Infective cystitis', 'Cold hands', 'Blurring of visual image', 'glibenclamide', 'Right ventricular dilatation', 'Flatulence/wind', 'Irritability and anger', 'Peripheral neuropathy', 'Intolerant of heat', 'Dyssomnia', 'Disorder of joint of shoulder region', 'Major depressive disorder', 'Solaraze', 'Pain radiating to lumbar region of back', 'magnesium', 'Eye strain', 'Tightness in throat', 'Mild depression', 'Impatient character', 'Miscarriage', 'Neurological symptom', 'Impaired bed mobility', 'Severe visual impairment', 'Malignant neoplasm of bone', 'Myalgia', 'Facial palsy', 'Spasm of back muscles', 'Neuralgia', 'Bursitis', 'Numbness of foot', 'Cold extremities', 'Memory impairment', 'Disability', 'Malignant tumour of prostate', 'Moody', 'Glossopyrosis', 'Joint swelling', 'Right hemiparesis', 'Ankle joint pain', 'Fear of falling', 'Tiredness symptom', 'Uterine spasm', 'Swollen ankle', 'Cramp in limb', 'Dizziness', 'Ulceration of gingivae', 'Difficulty standing up', 'Falls', 'Muscle twitch', 'Asthma', 'Cluster headache', 'Always hungry', 'Mastalgia', 'Autoimmune disease', 'Alopecia', 'Stammering', 'Liver enzymes abnormal', 'Acute gastritis', 'Transient memory loss', 'Celebrex', 'Flushing', 'zinc', 'Fibromyalgia', 'Thyroid disease', 'Pressure', 'Pain in pelvis', 'Obsessive-compulsive disorder', 'Renal impairment', 'Disturbance in physical behaviour', 'Muscle cramp', 'Gastrointestinal symptom', 'Intracranial tumour', 'Constantly crying', 'Type 1 diabetes mellitus', 'Generally unwell', 'Bleeding between periods', 'Necrosis of anatomical site', 'Oedema of hand', 'Pain in right arm', 'Hyperactive bowel sounds', 'Unable to initiate words', 'ezetimibe', 'Abdominal discomfort', 'Crestor', 'Breast tenderness', 'Pain in lower limb', 'Mood disorder', 'Pain in forearm', 'Mental distress', 'Angina', 'Advil', 'Shuffling gait', 'Gingival disease', 'Hunger pain', 'Pain in left leg', 'Increased liver function', 'Arthrotec', 'Migraine', 'Unable to control anger', 'amlodipine', 'Transient ischaemic attack', 'Cold feet', 'enalapril', 'Derealisation', 'Increased creatine kinase level', 'fish oil natural', 'Sharp pain', 'Fear', 'Raised serum calcium level', 'Urinary incontinence', 'Nausea', 'Ankle pain', 'Lumbar arthritis', 'Pain in buttock', 'Unable to get out of a chair', 'Delusional disorder', 'Mentally alert', 'Malignant neoplasm of brain', 'Hyperaemia of surface of eye', 'Buzzing in ear', 'Arthritis of knee', 'Rapid shallow breathing', 'Norvasc', 'Weight increased', 'Lipitor 20 mg tablet: film-coated', 'Loss of sense of smell', 'Not getting enough sleep', 'Lack of stamina', 'Encephalopathy', 'Middle insomnia', 'Joint stiffness', 'Suicidal', 'Bloating symptom', 'Tendonitis of foot', 'Spots on skin', 'Lescol', 'Endometriosis', 'Muscle weakness', 'Hand joint stiff', 'Abdominal bloating', 'Visual impairment', 'Acute renal failure', 'Increased belching', 'Irritable bowel syndrome', 'ascorbic acid', 'Erythema of skin', 'Epipen Auto-Injector', 'Constant pain', 'Transient global amnesia', 'Swelling of finger', 'Renal injury', 'Increased thirst', 'Proximal interphalangeal joint of finger pain', 'Agony', 'Tingling of skin', 'Sinusitis', 'Mobic', 'Sweating problem', 'Always sleepy', 'Watery eye', 'hydrochlorothiazide', 'Excessive sweating', 'Increased cholesterol esters', 'Musculoskeletal chest pain', 'ramipril', 'Jaw pain', 'Dark stools', 'Hypertension', 'Sacroiliac joint stiff', 'Shoulder girdle weakness', 'Pain in thumb', 'Sexual dysfunction', "Alzheimer's disease", "Parkinson's disease", 'Sudden visual loss', 'Difficulty chewing', 'Rhabdomyolysis', 'Sleep pattern disturbance', 'Skin irritation', 'Furuncle', 'Influenza-like symptoms', 'Pain in limb', 'Low blood pressure reading', 'Blepharospasm', 'Rhinitis', 'Heavy feeling in eyelids', 'Itching of skin', 'Trembles', 'Nightmares', 'Functional disorder of bladder', 'Ovarian pain', 'Pins and needles', 'Poor muscle tone', 'Shoulder stiff', 'Musculoskeletal pain', 'Joint crepitus', 'Hand pain', 'glipizide', 'Colitis', 'cyanocobalamin', 'Eruption', 'Menstrual loss increasing', 'Feeling hopeless', 'Naprosyn', 'Paralysis', 'Swelling of body region', 'Excessive weight loss', 'Mood swings', 'pseudoephedrine', 'Cerebrovascular accident', 'Daily headache', 'Lopid', 'Difficulty moving hand', 'Plantar fasciitis', 'Ulcer', 'Hip stiff', 'atenolol', 'Spasm', 'Eczema', 'Nonspecific abdominal symptom', 'Type II diabetes mellitus uncontrolled', 'Decreased progesterone level', 'Sweating', 'Hammer toe', 'Ehlers-Danlos syndrome', 'Choking', 'Bizarre dreams', 'Illness', 'Stomach ache', 'Nasonex Aqueous', 'Osteoarthritis of knee', 'Impending shock', 'Arthralgia of multiple joints', 'Feels hot', 'lysine', 'Dysmenorrhoea', 'Overweight', 'Temporal lobe epilepsy', 'Muscle tension', 'Sense of smell impaired', 'Anxiety attack', 'Palpitations', 'Heat stroke', 'Bad taste in mouth', 'simvastatin', 'Mentally vague', 'Diarrhoea', 'methylprednisolone', 'Gallbladder disorder', 'Loss of motivation', 'Gastric ulcer with perforation', 'Restless legs', 'Goitre', 'Excessive appetite', 'Gingival oedema', 'Blackout', 'Dislocation of joint', 'Malaria', 'Tightness in arm', 'Loss of taste', 'Chronic rhinitis', 'Joint tenderness', 'Muscle Cramp', 'Raised low density lipoprotein cholesterol', 'Avandia', 'Worried', 'Difficulty swallowing', 'Problem of visual accommodation', 'Intermittent pain', 'Decreased cholesterol esters', 'Body fluid retention', 'Unable to concentrate', 'Difficulty getting off a bed', 'Cramp in foot', 'Heel pain', 'Feeling bad', 'Strain of calf muscle', 'Clicking knee', 'Unsteady when walking', 'soy lecithin', 'Acne', 'Neuropathy of lower limb', 'Dysphonia', 'Finger joint locking', 'Blood clot in eye', 'Labour pain', 'Unable to balance', 'Oedema of lower extremity', 'Malaise', 'Heart disease', 'Kidney disease', 'Rotator cuff syndrome', 'Arthralgia of the upper arm', 'Muscle fatigue', 'Wrist joint pain', 'Slow on legs', 'Irregular menstruation', 'Sensation of hot and cold', 'Feeling unhappy', 'Polymyositis', 'Feeling empty', 'Hot sweats', 'Knee pain', 'Rib pain', 'Plavix', 'Language difficulty', 'Mild pain', 'Weakness of back', 'Cozaar', 'Pain in finger', 'Elbow stiff', 'Increased venous pressure', 'Low motivation', 'Sedated', 'Parkinsonian tremor', 'Intracranial aneurysm', 'Premarin', 'Nervousness', 'Lower abdominal pain', 'Polyarthritis associated with another disorder', 'Cataract', 'Trigeminal neuralgia', 'Heavy feeling', 'Unable to bend down', 'Ankle stiff', 'Inflammation of ligament', 'Sensitivity', 'Extreme exhaustion', 'Hand cramps', 'Urine looks dark', 'Blistering rash', 'Muscle injury', 'Disorder of eye', 'Bleeding', 'Abdominal pain', 'Intraocular haemorrhage', 'Joint pain', 'Tachycardia', 'Blood pressure alteration', 'Visual field scotoma', 'Arthralgia', 'Tightness sensation', 'Hungry', 'diclofenac + misoprostol', 'Ulcer of oesophagus', 'Dehydration', 'Stiff legs', 'Congestion of throat', 'Poor balance', 'Abnormal liver function', 'Irregular bowel habits', 'Hypochondriasis', 'Epigastric discomfort', 'Jaundice', 'Iliotibial band friction syndrome', 'Itching', 'Chronic diarrhoea', 'Olmetec', 'Fine motor impairment', 'Wrist stiff', 'aspirin', 'Bumping heart', 'Unable to sit', 'codeine', 'Bleeding from nose', 'Scalp itchy', 'Shoulder joint pain', 'White discolouration of skin', 'Loss of appetite', 'Poor self-image', 'Charleyhorse', 'Diabetes mellitus', 'pethidine', 'Moderate anxiety', 'Myalgia/myositis - forearm', 'Gastrointestinal tract problem', 'prednisone', 'Labyrinthitis', 'Unable to move all four limbs', 'Poor peripheral circulation', 'Unable to move arm', 'Chest discomfort', 'capsaicin', 'Coronary arteriosclerosis', 'Pain in elbow', 'olive oil', 'Efexor-XR', 'Anaemia', 'Rupture of Achilles tendon', 'Pounding heart', 'clonidine', 'Adhesive capsulitis of shoulder', 'Lipidil', 'Constipation', 'Retching', 'Aortic aneurysm', 'ibuprofen 5% gel', 'Metatarsalgia', 'Bed-ridden', 'Unsteady when standing', 'Ulceration of colon', 'Decreased liver function', 'Serum cholesterol raised', 'Insomnia', 'cortisone', 'Crying', 'Liver function tests abnormal', 'Aphthous ulcer', 'Disorder of muscle', 'Panic attack', 'Acenaesthesia', 'Amnesia', 'Pain in the coccyx', 'Photosensitivity', 'Skin tenderness', 'Difficulty writing', 'naproxen', 'Poor short-term memory', 'Cough', 'Carpal tunnel syndrome', 'Pancreatitis', 'Rupture of anterior cruciate ligament', 'Blister', 'Sciatica', 'Failure to lose weight', 'Haemorrhoids', 'Foot pain', 'Abdominal pressure', 'Arthritis', 'Allergic reaction', 'Hepatic trauma', 'Suicidal thoughts', 'Lethargy', 'Intervertebral disc prolapse', 'paracetamol + ibuprofen', 'Easy bruising', 'Hyperactive behaviour', 'Nexium', 'Poor concentration', 'Cardiomyopathy', 'Loss of confidence', 'Morbid thoughts', 'Difficulty breathing', 'Difficulty getting out of a chair', 'Cardiovascular disease', 'Zocor', 'Excessive day and night-time sleepiness', 'Tendon contracture', 'Hypersomnia', 'Vaginospasm', 'Blind or low vision - one eye only', 'Upset stomach', 'Swelling', 'Ophthalmic migraine', 'Motor neurone disease', 'Feeling lonely', 'Throbbing headache', 'Intra-abdominal haematoma', 'Pain', 'Tunnel vision', 'omeprazole', 'Viral upper respiratory tract infection', 'Arthralgia of the lower leg', 'Arthritis/arthrosis', 'Joint injury', 'Abdominal hyperaesthesia', 'Feeling high', 'potassium', 'Dropped beats - heart', 'Lack of libido', 'Lyrica', 'Feeling irritable', 'Haematochezia', 'pravastatin', 'Unable to feed self', 'Cholestatic jaundice syndrome', 'Foot joint stiff', 'Noise intolerance', 'alpha tocopherol', 'Cardiac arrhythmia', 'Dependance on walking stick', 'Joint laxity', 'Urine: dark/concentrated', 'Pain in calf', 'Elbow joint swelling', 'Paraesthesia of upper limb', 'Numbness of upper limb', 'Facial spasm', 'Multiple sclerosis', 'Initial insomnia', 'Chronic intractable pain', 'Muscle atrophy', 'Urinary tract infection', 'Large breast', 'Loss of voice', 'Renal calculus', 'Drowsy', 'Noten', 'Increased lactic acid level', 'Herpes zoster', 'Injury of tendon of the rotator cuff of shoulder', 'Steatosis of liver', 'Hip pain', 'Unable to lie down', 'Type 2 diabetes mellitus', 'Morning stiffness - joint', 'codeine + ibuprofen', 'Unable to make a fist', 'Hepatomegaly', 'Tenalgia', 'Lipitor', 'Disorder of skin', 'Involuntary movement', 'Influenza-like illness', 'Unable to get off a bed', 'Psychotic disorder', 'Testosterone', 'Indigestion', 'Feeling angry', 'Vertigo', 'Sinus pain', 'Numbness of lower limb', 'fenofibrate', 'Stevens-Johnson syndrome', 'Nasal sinus problem', 'Myalgia/myositis - upper arm', 'Loss of hair', 'Neuropathy', 'Dysfunctional uterine bleeding', 'Lyme disease', 'pyridoxine', 'Glossitis', 'Fragile self-esteem', 'Arthritis of hip', 'Vomiting', 'Stiff neck', 'Restlessness', 'Blood coagulation disorder', 'Heavy episode of vaginal bleeding', 'Unable to move leg', 'Peritonitis', 'White blood cell count abnormal', 'Weakness of neck', 'Decrease in appetite', 'Exhaustion', 'Unable to run', 'Bacterial sepsis', 'Disorder of vision', 'Paraesthesia', 'Abdominal swelling', 'Mucosal numbness', 'Osteoarthrosis of hand', 'Loss of equilibrium', 'Stress', 'Contusion of hand', 'Nerve injury', 'Scleral icterus', 'Injury of muscle', 'Toothache', 'Infectious disease of lung', 'Stomach problem', 'Haematuria', 'Hypomenorrhoea', 'Unable to walk up step', 'Abdominal distension, gaseous', 'Anger reaction', 'Excruciating pain', 'Pain in throat', 'Pruritus ani', 'Traumatic blister of mouth', 'Myopathy', 'Numbness of finger', 'Generalised pruritus', 'Nocturnal enuresis', 'Abdominal wind pain', 'Deteriorating renal function', 'Muscle rigidity', 'Difficulty standing', 'Burning sensation of vagina', 'Asthenia', 'Ezetrol', 'Common cold', 'Rupture of muscle', 'Rhinocort Aqueous', 'caffeine', 'Loss of interest', 'Tylenol', 'Tired', 'Blurred vision - hazy', 'glucosamine', 'Severe depression', 'Bronchitis', 'Limitation of joint movement', 'Vaginal bleeding', 'Tinea pedis', 'Difficulty lifting', 'Feeling faint', 'Night sweats', 'Burning pain', 'Aleve', 'Abrasion of eye region', 'Confusion', 'Bursitis of hip', 'Degenerative disease of the central nervous system', 'Thigh pain', 'Paraesthesia of lower extremity', 'Basic learning problem', 'Abnormal muscle function', 'Chest pain', 'Liver disease', 'Hepatic failure', 'Cymbalta', 'Paraesthesia of foot', 'Neck pain', 'Dry hair', 'Myositis', 'Thoughts of self harm', 'Respiratory depression', 'Irritable bowel syndrome characterised by constipation', "Bell's palsy", 'Increased lipid', 'Dry eyes', 'Swollen knee', 'Facial swelling', 'Lexapro', 'Flank pain', 'Decreased testosterone level', 'Hives', 'Malignant tumour of colon', 'Digestive symptom', 'Pain in toe', 'Voltaren', 'C/O - feeling depressed', 'Burning feet', 'Mental disorder', 'Unable to climb stairs', 'Tires quickly', 'Backache', 'Reflux', 'Abnormal blood pressure', 'Calcaneal spur', 'Gingivitis', 'Excitement', 'Menstrual spotting', 'Tired all the time', 'Bowel dysfunction']
    cat_to_id=dict(zip(categories,range(len(categories))))
    return categories,cat_to_id

def process_file(filename,word_to_id,cat_to_id,max_length=200):
    """
    Args:
        filename:train_filename or test_filename or val_filename
        word_to_id:get from def read_vocab()
        cat_to_id:get from def read_category()
        max_length:allow max length of sentence 
    Returns:
        x_pad: sequence data from  preprocessing sentence 
        y_pad: sequence data from preprocessing label

    """
    labels,contents=read_file(filename)
    data_id,label_id=[],[]
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])
    x_pad=kr.preprocessing.sequence.pad_sequences(data_id,max_length,padding='post', truncating='post')
    y_pad=kr.utils.to_categorical(label_id)
    return x_pad,y_pad

def batch_iter(x,y,batch_size=64):
    """
    Args:
        x: x_pad get from def process_file()
        y:y_pad get from def process_file()
    Yield:
        input_x,input_y by batch size

    """

    data_len=len(x)
    num_batch=int((data_len-1)/batch_size)+1

    indices=np.random.permutation(np.arange(data_len))
    x_shuffle=x[indices]
    y_shuffle=y[indices]

    for i in range(num_batch):
        start_id=i*batch_size
        end_id=min((i+1)*batch_size,data_len)
        yield x_shuffle[start_id:end_id],y_shuffle[start_id:end_id]

def export_word2vec_vectors(vocab, word2vec_dir,trimmed_filename):
    """
    Args:
        vocab: word_to_id 
        word2vec_dir:file path of have trained word vector by word2vec
        trimmed_filename:file path of changing word_vector to numpy file
    Returns:
        save vocab_vector to numpy file
        
    """
    file_r = codecs.open(word2vec_dir, 'r', encoding='utf-8')
    line = file_r.readline()
    voc_size, vec_dim = map(int, line.split(' '))
    embeddings = np.zeros([len(vocab), vec_dim])
    line = file_r.readline()
    while line:
        try:
            items = line.split(' ')
            word = items[0]
            vec = np.asarray(items[1:], dtype='float32')
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(vec)
        except:
            pass
        line = file_r.readline()
    np.savez_compressed(trimmed_filename, embeddings=embeddings)

def get_training_word2vec_vectors(filename):
    """
    Args:
        filename:numpy file
    Returns:
        data["embeddings"]: a matrix of vocab vector
    """
    with np.load(filename) as data:
        return data["embeddings"]


def get_sequence_length(x_batch):
    """
    Args:
        x_batch:a batch of input_data
    Returns:
        sequence_lenghts: a list of acutal length of  every senuence_data in input_data
    """
    sequence_lengths=[]
    for x in x_batch:
        actual_length = np.sum(np.sign(x))
        sequence_lengths.append(actual_length)
    return sequence_lengths