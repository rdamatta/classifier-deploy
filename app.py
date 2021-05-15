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
    image_camp = Image.open('oil.png')

    #st.image(image,use_column_width=False)

    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))

    st.sidebar.info('This app was created to predict oil samples evaluation.')
    
    st.sidebar.image(image_camp)

    st.title("Oil Prediction App")

    if add_selectbox == 'Online':

        input1 = st.selectbox('Make and Type', ['1-1', '1-2', '1-3', '2-1', '2-2'])
        input2 = st.selectbox('Grade', ['15W-40', '30', '68', '75W-90', '85W-140'])
        input3 = st.number_input('Machine Hours (oil)', min_value=0, max_value=9000, value=660)
        input4 = st.number_input('Machine Hours (equip.)', min_value=0, max_value=12000, value=7600)
        input5 = st.number_input('PQI', min_value=0, max_value=716, value=35)
        input6 = st.selectbox('Iron (Fe)', ['<1','1 to 192', '192 to 383', '383 to 574', '574 to 764'])
        input7 = st.selectbox('Copper (Cu)', ['<1','1 to 71', '141 to 211', '211 to 282', '71 to 141'])
        input8 = st.selectbox('Chrome (Cr)', ['<1','1', '2', '3', '4', '5', '6','7', '8', '9'])
        input9 = st.selectbox('Lead (Pb)', ['<1','1 to 17', '17 to 33', '33 to 49', '49 to 67'])
        input10 = st.selectbox('Nickel (Ni)', ['<1','1 to 7', '13 to 19', '19 to 24', '7 to 13'])
        input11 = st.selectbox('Aluminium (Al)', ['<1','1 to 37', '109 to 147'])
        input12 = st.selectbox('Silicon (Si)', ['<1','1 to 12', '12 to 23', '23 to 34', '34 to 44'])
        input13 = st.selectbox('Sodium (Na)', ['<1','1 to 98', '195 to 292', '292 to 388', '98 to 195'])
        input14 = st.selectbox('Potassium (K)', ['<1','1', '2', '3', '4', '5', '6', '8', '9', '10','11','12'])
        input15 = st.selectbox('Molybdenum (Mo)', ['<1','1 to 15', '15 to 29', '29 to 43', '43 to 58'])
        input16 = st.selectbox('Boron (B)', ['<1','1 to 132', '132 to 263', '263 to 394', '394 to 525'])
        input17 = st.selectbox('Barium (Ba)', ['<1','1', '2', '3'])
        input18 = st.selectbox('Magnesium (Mg)', ['<1','1 to 286', '286 to 571', '571 to 856', '856 to 1140'])
        input19 = st.selectbox('Calcium (Ca)', ['<1','1 to 1129', '1129 to 2257', '2257 to 3385', '3385 to 4514'])
        input20 = st.selectbox('Zinc (Zn)', ['<1','1 to 366', '1096 to 1462', '366 to 731', '731 to 1096'])
        input21 = st.number_input('Phosphorus (P)', min_value=0, max_value=1, value=1)
        input22 = st.selectbox('Silver (Ag)', ['<1','1', '2', '3','4', '5','6', '7', '8', '9', '10','11', '12', '13', '14', '15', '16'])
        input23 = st.selectbox('Manganese (Mn)', ['<1','1', '2', '3', '4', '5', '6'])
        input24 = st.selectbox('Vanadium (V)', ['<1','1'])
        input25 = st.selectbox('Titanium (Ti)', ['<1','1'])
        input26 = st.selectbox('Cadmium (Cd)', ['<1','1','2', '3'])
        input27 = st.selectbox('Fluor (F)', ['N', 'P'])
        input28 = st.number_input('Viscosity at 40°C', min_value=60, max_value=400, value=207)
        input29 = st.number_input('Water Content', min_value=0.0, max_value=1.0, value=0.01)
        input30 = st.number_input('Oxide Content', min_value=0, max_value=80, value=16)
        input31 = st.number_input('Nitrate Content', min_value=0, max_value=20, value=5)
        input32 = st.number_input('Sulphate Content', min_value=0, max_value=100, value=23)
        input33 = st.selectbox('ISO-6 Cleanliness', ['-', '0 to 6', '12 to 18', '18 to 23', '6 to 12'])
        input34 = st.selectbox('ISO-14 Cleanliness', ['-', '0 to 6', '12 to 18', '18 to 22', '6 to 12'])
        input35 = st.selectbox('>6μ Particle Count', ['-','0 to 15093', '15093 to 30186', '30186 to 45279','45279 to 60372'])
        input36 = st.selectbox('>10μ Particle Count', ['-', '0 to 9806', '19612 to 29418', '29418 to 39225','9806 to 19612'])
        input37 = st.selectbox('>14μ Particle Count', ['-', '0 to 7782', '15564 to 23346', '23346 to 31127','7782 to 15564'])
        input38 = st.selectbox('>21μ Particle Count', ['-', '0 to 4462', '13386 to 17848', '4462 to 8924','8924 to 13386'])
        input39 = st.selectbox('>25μ Particle Count', ['-', '0 to 1613', '1613 to 3226', '3226 to 4839', '4839 to 6452'])
        input40 = st.selectbox('>38μ Particle Count', ['-', '0 to 145', '145 to 290', '290 to 435', '435 to 581'])
        input41 = st.selectbox('>70μ Particle Count', ['-', '0 to 16', '16 to 32', '48 to 62'])


        output=""

        input_dict = {'oiltypeid' : input1, 'oilgradeid' : input2, 'oilhours' : input3, 'machinehours' : input4, 'PQI' : input5, 'Fe' : input6, 'Cu' : input7, 'Cr' : input8, 'Pb' : input9, 'Ni' : input10, 'Al' : input11, 'Si' : input12, 'Na' : input13, 'K' : input14, 'Mo' : input15, 'B' : input16, 'Ba' : input17, 'Mg' : input18,'Ca' : input19,'Zn' : input20,'P' : input21,'Ag' : input22,'Mn' : input23,'V' : input24,'Ti' : input25,'Cd' : input26,'F' : input27, 'V40' : input28, 'H2O' : input29,'OXI' : input30,'NIT' : input31,'SUL' : input32,'ISO6' : input33,'ISO14' : input34,'X6' : input35,'X10' : input36,'X14' : input37,'X21' : input38,'X25' : input39,'X38' : input40,'X70' : input41}
        input_df = pd.DataFrame([input_dict])

        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
        if output == 'A':
            st.success("Oil properties are within acceptable limits and operation can continue as usual.")
        elif output == 'B':
            st.success("Certain results are outside acceptable ranges, minor problems with machinery.")
        elif output == 'C':
            st.success("Unsatisfactory results are present, significant problem with the compartment and lubricant properties.")
        elif output == 'X':
            st.success("Clear contamination needing immediate diagnostic and corrective action to prevent possible failure.")

        #st.success('The outcome is: {}'.format(output))

    if add_selectbox == 'Batch':

        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model,data=data)
            st.write(predictions)

if __name__ == '__main__':
    run()
