import codecs
from tkinter import *
from tkinter import font
from tkinter.filedialog import askopenfilename

from matplotlib import pyplot as plt

from Test import bar_plot
from model_4_lbl import __4_lbl
from model_5_lbl import __5_lbl

WindowTitle = 'Arabic Word Segmenter'
completeModel_5_lbl, completeModel_4lbl = False, False
accuracyLabel_5_lbl, accuracyLabel_4_lbl = [], []
accuracy_5_lbl, accuracy_4_lbl = 0, 0

# -------------------------------------------------- segmenter settings --------------------------------
_4_ca_setting = {
    'max_len': 80,
    'embed_dim': 200,
    'lstm_dim': 200,
    'learn_rate': 0.01,
}
_5_ca_setting = {
    'max_len': None,
    'embed_dim': 200,
    'lstm_dim': 200,
    'learn_rate': 0.01,
}

model_4__ca = __4_lbl('CA', _4_ca_setting)
model_5__ca = __5_lbl('CA', _5_ca_setting)


# -------------------------------------------------- widget settings --------------------------------

def clear_input():
    input_text.delete(1.0, "end")
    output_text_4.configure(state=NORMAL)
    output_text_5.configure(state=NORMAL)
    output_text_4.delete(1.0, "end")
    output_text_5.delete(1.0, "end")
    output_text_4.configure(state=DISABLED)
    output_text_5.configure(state=DISABLED)


def loadFile():
    filename = askopenfilename(title="Select file", filetypes=(("Text Files", "*.txt"), ("All", "*.*")))
    with codecs.open(filename, encoding='utf-8') as fl:
        for line in fl:
            input_text.insert('end', line.strip() + '\n')


def seg_input():
    text_in = input_text.get(1.0, 'end')
    if meth.get() == 2:
        output_text_4.configure(state=NORMAL)
        output = model_4__ca.segment_input(text_in)
        output_text_4.insert('end', output + '\n')
        output_text_4.configure(state=DISABLED)
    else:
        output_text_5.configure(state=NORMAL)
        output = model_5__ca.segment_input(text_in)
        output_text_5.insert('end', output + '\n')
        output_text_5.configure(state=DISABLED)


def comparison():
    data = {
        'Met 4 lbl': [98.77, 96.40, 98.99, 98.8, 100],
        'Met 5 lbl': [96.95, 92.73, 97.48, 100, 96.66]
    }
    data_2 = {
        'Met 4 lbl': [98.64, 97.24],
        'Met 5 lbl': [97.29, 92.65]
    }
    labels = ['', 'label B', 'label M', 'label E', 'label S', 'label WB']
    labels_2 = ['', '', 'Char-based', '', '', '', 'Word-based']
    fig, ax = plt.subplots()
    _, ax_2 = plt.subplots()
    plt.ylim(0, 120)
    ax.set_ylabel('Percentage')
    ax_2.set_ylabel('Percentage')
    ax.set_title('Percentage by Labels for each methods')
    ax_2.set_title('Percentage by mode-based for each methods')
    ax.set_xticklabels(labels)
    ax_2.set_xticklabels(labels_2)
    bar_plot(ax, data, total_width=.8, single_width=.9)
    bar_plot(ax_2, data_2, total_width=.8, single_width=.9)
    plt.grid(True)
    plt.show()


root = Tk()
wrapper_option = LabelFrame(root, text='Options', bg='white')
wrapper_input = LabelFrame(root, text='Input', bg='white')
wrapper_output_5 = LabelFrame(root, text='Output 5 lbl', bg='white')
wrapper_output_4 = LabelFrame(root, text='Output 4 lbl', bg='white')
#
wrapper_option.place(x=5, y=0, width=1070, height=150)
wrapper_input.place(x=5, y=155, width=1070, height=210)
wrapper_output_5.place(x=5, y=370, width=530, height=225)
wrapper_output_4.place(x=545, y=370, width=530, height=225)
# ------------- content input----------
myFont = font.Font(family='calibri', size=15)
myFont_2 = font.Font(family='Cambria', weight='bold', size=12)
input_text = Text(wrapper_input, relief=RIDGE, borderwidth=2, font=myFont)
btn_seg_text = Button(wrapper_input, text='Segment', bg='#256ef4', fg='white', padx=10, font=myFont,
                      command=seg_input)
btn_load_file = Button(wrapper_input, text='Load file', bg='#c53636', fg='white', padx=10, font=myFont,
                       command=loadFile)
btn_clear = Button(wrapper_input, text='Clear', bg='white', fg='black', padx=10, font=myFont,
                   command=clear_input)
btn_ev = Button(wrapper_input, text='Comparison', bg='#5330a8', fg='white', font=myFont,
                command=comparison)

input_text.place(x=5, y=0, width=920, height=185)
btn_seg_text.place(x=940, y=0, width=110, height=35)
btn_load_file.place(x=940, y=45, width=110, height=35)
btn_clear.place(x=940, y=90, width=110, height=35)
btn_ev.place(x=940, y=140, width=110)
# ------------- content output----------
output_text_5 = Text(wrapper_output_5, bg='white', relief=RIDGE, borderwidth=2, font=myFont)
output_text_4 = Text(wrapper_output_4, bg='white', relief=RIDGE, borderwidth=2, font=myFont)
output_text_5.pack(expand=1, fill="both", padx=5, pady=5)
output_text_4.pack(expand=1, fill="both", padx=5, pady=5)
output_text_5.configure(state="disabled")
output_text_4.configure(state="disabled")
# model settings

# options 220x240
lbl_title = Label(wrapper_option, text="""                                    THESIS
                                    Presented for graduation from
                                    MASTER IN COMPUTER SCIENCES
                                    Through
                                    D I N A R       A B D E L A Z I Z
                                    ---- Arabic Word Segmentation ----""", bg='white', font=myFont_2)
wrapper_model = LabelFrame(wrapper_option, text='Model Selected', bg='white')
lbl_title.place(x=0, y=0, width=850, height=130)
wrapper_model.place(x=860, y=0, width=200, height=130)
# ------------- content ----------
meth = IntVar()
meth.set(2)
lbl_r_met_5lbl = Label(wrapper_model, text='Method 5 lbls', bg='white')
lbl_r_met_4lbl = Label(wrapper_model, text='Method 4 lbls', bg='white')
__meth_1 = Radiobutton(wrapper_model, variable=meth, value=1, bg='white')
__meth_2 = Radiobutton(wrapper_model, variable=meth, value=2, bg='white')

lbl_r_met_5lbl.place(x=55, y=10)
lbl_r_met_4lbl.place(x=55, y=50)
__meth_1.place(x=10, y=10)
__meth_2.place(x=10, y=50)

#


root.title(WindowTitle)
root.geometry("1080x600")
# lance tkinter
root.config(bg='white')
root.mainloop()
