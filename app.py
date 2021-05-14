from pycaret.classification import *
from imblearn.over_sampling import *
import streamlit as st
import pandas as pd
import numpy as np
import pickle

model = load_model('deployment_14052021')

def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions

def run():

    from PIL import Image
    image_camp = Image.open('oil.jpeg')

    #st.image(image,use_column_width=False)

    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))

    st.sidebar.info('This app was created to predict oil samples evaluation.')
    
    st.sidebar.image(image_camp)

    st.title("Oil Prediction App")

    if add_selectbox == 'Online':

        input1 = st.selectbox('Serial #', [1,2,3,4,5,6,7,8])
        input2 = st.selectbox('Compartiment ID', [107, 118, 146, 172, 317, 326, 364, 422, 423, 424, 425, 432, 433])
        input3 = st.selectbox('Make and Type', ['1-1', '1-2', '1-3', '2-1', '2-2'])
        input4 = st.selectbox('Grade', ['15W-40', '30', '68', '75W-90', '85W-140'])
        input5 = st.number_input('Machine Hours(oil)', min_value=0, max_value=9000, value=660)
        input6 = st.number_input('Machine Hours(equip)', min_value=0, max_value=12000, value=7600)
        input7 = st.number_input('PQI', min_value=0, max_value=716, value=35)
        input8 = st.selectbox('Iron', ['<1','1 to 192', '192 to 383', '383 to 574', '574 to 764'])
        input9 = st.selectbox('Copper', ['<1','1 to 71', '141 to 211', '211 to 282', '71 to 141'])
        input10 = st.selectbox('Chrome', ['<1','1', '2', '3', '4', '5', '6','7', '8', '9'])
        input11 = st.selectbox('Lead', ['<1','1 to 17', '17 to 33', '33 to 49', '49 to 67'])
        input12 = st.selectbox('Nickel', ['<1','1 to 7', '13 to 19', '19 to 24', '7 to 13'])
        input13 = st.selectbox('Aluminium', ['<1','1 to 37', '109 to 147'])
        input14 = st.selectbox('Silicon', ['<1','1 to 12', '12 to 23', '23 to 34', '34 to 44'])
        input15 = st.selectbox('Sodium', ['<1','1 to 98', '195 to 292', '292 to 388', '98 to 195'])
        input16 = st.selectbox('Potassium', ['<1','1', '2', '3', '4', '5', '6', '8', '9', '10','11','12'])
        input17 = st.selectbox('Molybdenum', ['<1','1 to 15', '15 to 29', '29 to 43', '43 to 58'])
        input18 = st.selectbox('Boron', ['<1','1 to 132', '132 to 263', '263 to 394', '394 to 525'])
        input19 = st.selectbox('Barium', ['<1','1', '2', '3'])
        input20 = st.selectbox('Magnesium', ['<1','1 to 286', '286 to 571', '571 to 856', '856 to 1140'])
        input21 = st.selectbox('Calcium', ['<1','1 to 1129', '1129 to 2257', '2257 to 3385', '3385 to 4514'])
        input22 = st.selectbox('Zinc', ['<1','1 to 366', '1096 to 1462', '366 to 731', '731 to 1096'])
        input23 = st.number_input('Phosphorus', min_value=0, max_value=1, value=1)
        input24 = st.selectbox('Silver', ['<1','1', '2', '3','4', '5','6', '7', '8', '9', '10','11', '12', '13', '14', '15', '16'])
        input25 = st.selectbox('Manganese', ['<1','1', '2', '3', '4', '5', '6'])
        input26 = st.selectbox('Vanadium', ['<1','1'])
        input27 = st.selectbox('Titanium', ['<1','1'])
        input28 = st.selectbox('Cadmium', ['<1','1','2', '3'])
        input29 = st.number_input('Water Content', min_value=0.0, max_value=1.0, value=0.01)
        input30 = st.selectbox('Fluor', ['N', 'P'])
        input31 = st.number_input('Viscosity', min_value=60, max_value=400, value=207)
        input32 = st.number_input('Oxide Content', min_value=0, max_value=80, value=16)
        input33 = st.number_input('Nitrate Content', min_value=0, max_value=20, value=5)
        input34 = st.number_input('Sulphate Content', min_value=0, max_value=100, value=23)
        input35 = st.selectbox('Cleanliness(ISO-6)', ['-', '0 to 6', '12 to 18', '18 to 23', '6 to 12'])
        input36 = st.selectbox('Cleanliness(ISO-14)', ['-', '0 to 6', '12 to 18', '18 to 22', '6 to 12'])
        input37 = st.selectbox('Particle count >6μ', ['-','0 to 15093', '15093 to 30186', '30186 to 45279','45279 to 60372'])
        input38 = st.selectbox('Particle count >10μ', ['-', '0 to 9806', '19612 to 29418', '29418 to 39225','9806 to 19612'])
        input39 = st.selectbox('Particle count >14μ', ['-', '0 to 7782', '15564 to 23346', '23346 to 31127','7782 to 15564'])
        input40 = st.selectbox('Particle count >21μ', ['-', '0 to 4462', '13386 to 17848', '4462 to 8924','8924 to 13386'])
        input41 = st.selectbox('Particle count >25μ', ['-', '0 to 1613', '1613 to 3226', '3226 to 4839', '4839 to 6452'])
        input42 = st.selectbox('Particle count >38μ', ['-', '0 to 145', '145 to 290', '290 to 435', '435 to 581'])
        input43 = st.selectbox('Particle count >70μ', ['-', '0 to 16', '16 to 32', '48 to 62'])


        output=""

        input_dict = {'serialno' : input1, 'compartid' : input2, 'oiltypeid' : input3, 'oilgradeid' : input4, 'oilhours' : input5, 'machinehours' : input6, 'PQI' : input7, 'Fe' : input8, 'Cu' : input9, 'Cr' : input10, 'Pb' : input11, 'Ni' : input12, 'Al' : input13, 'Si' : input14, 'Na' : input15, 'K' : input16, 'Mo' : input17, 'B' : input18, 'Ba' : input19, 'Mg' : input20,'Ca' : input21,'Zn' : input22,'P' : input23,'Ag' : input24,'Mn' : input25,'V' : input26,'Ti' : input27,'Cd' : input28,'H2O' : input29,'F' : input30,'V40' : input31,'OXI' : input32,'NIT' : input33,'SUL' : input34,'ISO6' : input35,'ISO14' : input36,'X6' : input37,'X10' : input38,'X14' : input39,'X21' : input40,'X25' : input41,'X38' : input42,'X70' : input43}
        input_df = pd.DataFrame([input_dict])

        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
        if output == 0:
            res = "Oil properties are within acceptable limits and operation can continue as usual."
        if output == 1:
            res = "Certain results are outside acceptable ranges, minor problems with machinery."
        if output == 2:
            res = "Unsatisfactory results are present, significant problem with the compartment and lubricant properties."
        if output == 3:
            res = "Clear contamination needing immediate diagnostic and corrective action to prevent possible failure."

        st.success(res)

    if add_selectbox == 'Batch':

        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model,data=data)
            st.write(predictions)

if __name__ == '__main__':
    run()
