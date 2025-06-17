# Importing libraries
import bw2io as bi
import bw2data as bd
import pandas as pd

def import_excel_database_to_brightway(data, sheet):
    # Save the data to a temporary file that can be used by ExcelImporter
    temp_path = r'C:\Users\ruw\Desktop\RA\N2O project\Data\databases\temp.xlsx'
    data.to_excel(temp_path, index=False)
    
    # Use the temporary file with ExcelImporter
    imp = bi.ExcelImporter(temp_path)  # the path to your inventory excel file
    imp.apply_strategies()

    # Loop through each database and match
    print(f"Matching database: {sheet}")
    imp.match_database(sheet, fields=('name', 'unit', 'location', 'reference product'))
    print(f"Unlinked items after matching {sheet}: {list(imp.unlinked)}")

    # Match without specifying a database
    imp.match_database(fields=('name', 'unit', 'location'))

    # Generate statistics and write results
    imp.statistics()
    imp.write_excel(only_unlinked=True)
    unlinked_items = list(imp.unlinked)
    imp.write_database()

    # Print unlinked items if needed
    print(unlinked_items)
    print(f'{data.columns[1]} is loaded into the database')
    import_excel_database_to_brightway.has_been_called = True

import_excel_database_to_brightway.has_been_called = False

def reload_database(sheet_name, system_path, reload=False):
    reload_database.has_been_called = True
    
    if reload:
        # Reload all databases
        for case, path in enumerate(system_path):    
            db_path = path 
            for db in sheet_name:
                # Removing the old database
                data = pd.read_excel(db_path, sheet_name=db)
                proj_database = data.columns[1]
                db_old = bd.Database(proj_database)
                db_old.deregister()
                
                import_excel_database_to_brightway(data, db)



reload_database.has_been_called = False

def database_setup(ecoinevnt_paths, system_path, bw_project="Single Use vs Multi Use", sheet_names = ['ev391consq', 'ev391cutoff'], reload=False):
    import_excel_database_to_brightway.has_been_called = False
    # Set the current Brightway project
    bd.projects.set_current(bw_project)

    # Check if biosphere database is already present
    if any("biosphere" in db for db in bd.databases):
        pass
    else:
        bi.bw2setup()

    # Check if Ecoinvent databases are already present
    if 'ev391consq' in bd.databases or 'ev391cutoff' in bd.databases:# and 'ev391apos' in bd.databases and 'ev391cutoff' in bd.databases:
        pass
    else:
        # # Import APOS database
        # ei = bi.SingleOutputEcospold2Importer(dirpath=ecoinevnt_paths['ev391apos'], db_name='ev391apos')
        # ei.apply_strategies()
        # ei.statistics()
        # ei.write_database()

        # Import Consequential database
        ei = bi.SingleOutputEcospold2Importer(dirpath=ecoinevnt_paths['ev391consq'], db_name='ev391consq')
        ei.apply_strategies()
        ei.statistics()
        ei.write_database()

        # Import Cut-off database
        ei = bi.SingleOutputEcospold2Importer(dirpath=ecoinevnt_paths['ev391cutoff'], db_name='ev391cutoff')
        ei.apply_strategies()
        ei.statistics()
        ei.write_database()

    # Import Excel databases
    for path in system_path:    
        for sheet in sheet_names:
            # Read the Excel file
            data = pd.read_excel(path, sheet_name=sheet)
            db = data.columns[1]
            if db not in bd.databases:
                import_excel_database_to_brightway(data, sheet)
    
    # Reload databases if needed
    if import_excel_database_to_brightway.has_been_called is False:
        reload_database(sheet_names, system_path, reload=reload)