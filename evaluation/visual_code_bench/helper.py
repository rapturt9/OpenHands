import os


def apply_changes_to_html(index_html, changes_md):
    """Apply changes from changes.md to index.html"""
    with open(index_html, 'a') as html_file, open(changes_md, 'r') as changes_file:
        changes = changes_file.read()
        html_file.write(f'\n<!-- Changes Applied: {changes} -->')


def compare_results(model_answer, final_ans):
    """Dummy comparison function"""
    return model_answer.strip() == final_ans.strip()


def create_sh_file(filename, cmds):
    """Creates a shell script file"""
    with open(filename, 'w') as file:
        file.write(cmds.replace('\r\n', '\n'))
    os.chmod(filename, 0o755)
