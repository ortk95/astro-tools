"""
Module for dealing with reference information.
"""
import bibtexparser


def load_bib(path):
    """
    Load data from bibtex file.

    Parameters
    ----------
    path : str
        Path to bibtex file.

    Returns
    -------
    dict containing bibtex file information.
    """
    with open(path, 'r') as f:
        db = bibtexparser.load(f)
    return db.entries_dict


def get_author_str(db, et_al=3):
    """
    Produce author string from a bibtex file.

    Parameters
    ----------
    db : dict
        load_bib() result for specific reference.

    et_al : int
        Number of authors to allow before 'et al.'

    Returns
    -------
    str
    """
    authors = db['author'].split(' and ')
    authors = [a.split(',')[0] for a in authors]
    if len(authors) > et_al:
        authors = authors[0] + ' et al.'
    elif len(authors) <= 2:
        authors = ' and '.join(authors)
    else:
        authors = ', '.join(authors[:-1]) + ' and ' + authors[-1]
    return authors
