import click
from pipeline import process

@click.command()

@click.argument('leftcsv', type=click.Path(exists=True))
@click.argument('rightcsv', type=click.Path(exists=True))
@click.option('--similiarity', default=0.8, help='Minimum similarity score (0-1)')
@click.option('--ntop', default=10, help='Top matches per record')
def run(leftcsv, rightcsv, similiarity, ntop):
    """CLI app to identify likely string matches of two csv files"""
    process(leftcsv, rightcsv, similiarity, ntop)

if __name__ == '__main__':
    run()